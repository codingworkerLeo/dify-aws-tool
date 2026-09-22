import functools
import json
import logging
import operator
import threading
import time
from typing import Any, Optional

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore

from dify_plugin import RerankModel
from dify_plugin.entities.model import AIModelEntity, FetchFrom, I18nObject, ModelType
from dify_plugin.entities.model.rerank import RerankDocument, RerankResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

logger = logging.getLogger(__name__)

# Sentinel for "argument not supplied" in _refresh_token(); distinct from an explicit None.
_UNSET = object()

# botocore ClientError codes that mean the caller's identity/credentials were rejected.
_AUTH_ERROR_CODES = frozenset({
    "AccessDenied",
    "AccessDeniedException",
    "UnauthorizedOperation",
    "ExpiredToken",
    "ExpiredTokenException",
    "InvalidClientTokenId",
    "SignatureDoesNotMatch",
    "UnrecognizedClientException",
    "AuthFailure",
})

# botocore ClientError codes that mean the request was throttled. This is the set of standard
# throttling codes botocore itself treats as retryable (botocore/data/_retry.json and
# botocore.retries.standard), written out explicitly so no private botocore attribute is imported.
_RATE_LIMIT_ERROR_CODES = frozenset({
    "Throttling",
    "ThrottlingException",
    "ThrottledException",
    "RequestThrottledException",
    "RequestThrottled",
    "TooManyRequestsException",
    "RequestLimitExceeded",
    "ProvisionedThroughputExceededException",
    "LimitExceededException",
    "BandwidthLimitExceeded",
    "SlowDown",
})


class SageMakerRerankModel(RerankModel):
    """
    Model class for SageMaker rerank model.
    """

    sagemaker_client: Any = None
    sagemaker_endpoint: str | None = None
    access_key: str = None
    secret_key: str = None
    aws_region: str = None
    assume_role_arn: str = None

    # Guards the compare-and-read and the publish of the six cached attributes above (two short
    # critical sections, no I/O), so that two threads sharing one model instance can never pair one
    # tenant's client with another's endpoint. This is a plain class attribute, i.e. one lock shared
    # by every instance of this class; client construction (including the STS AssumeRole round-trip)
    # deliberately happens OUTSIDE it, see _build_client(), so a slow or failing AssumeRole for one
    # tenant never blocks another tenant's cache hit.
    _client_lock: Any = threading.Lock()

    def _sagemaker_rerank(self, sm_client, query_input: str, docs: list[str], rerank_endpoint: str):
        inputs = [query_input] * len(docs)
        response_model = sm_client.invoke_endpoint(
            EndpointName=rerank_endpoint,
            Body=json.dumps({"inputs": inputs, "docs": docs}),
            ContentType="application/json",
        )
        json_str = response_model["Body"].read().decode("utf8")
        json_obj = json.loads(json_str)
        scores = json_obj["scores"]
        return scores if isinstance(scores, list) else [scores]

    def _refresh_token(self, access_key=_UNSET, secret_key=_UNSET, region=_UNSET, role_arn=_UNSET):
        """Refresh tokens by calling assume_role again.

        The STS call is made with the *source account* identity passed in as an explicit
        snapshot (access_key/secret_key/region/role_arn). ``_invoke`` binds that snapshot with
        ``functools.partial`` so the first AssumeRole and every automatic refresh use the same
        identity, independent of the published ``self.*`` state (which may already belong to a
        different tenant). Any argument that is not supplied falls back to ``self.*`` so the
        legacy no-argument call keeps working. An explicit ``None`` is honoured as "no value".
        """
        if access_key is _UNSET:
            access_key = self.access_key
        if secret_key is _UNSET:
            secret_key = self.secret_key
        if region is _UNSET:
            region = self.aws_region
        if role_arn is _UNSET:
            role_arn = self.assume_role_arn

        params = {
            "RoleArn": role_arn,
            "DurationSeconds": 3600,
            "RoleSessionName": f"dify-sagemaker-rerank-{int(time.time())}"
        }

        # Source-account session for STS: explicit AK/SK when configured, otherwise the runtime
        # environment's default credential chain (backward compatible).
        session_kwargs = {}
        if region:
            session_kwargs["region_name"] = region
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
        boto_session = boto3.Session(**session_kwargs)
        sts_client = boto_session.client("sts")

        response = sts_client.assume_role(**params).get("Credentials")

        credentials = {
            "access_key": response.get("AccessKeyId"),
            "secret_key": response.get("SecretAccessKey"),
            "token": response.get("SessionToken"),
            "expiry_time": response.get("Expiration").isoformat(),
        }

        return credentials

    def _build_client(self, access_key, secret_key, region, role_arn):
        """Build a sagemaker-runtime client from an immutable snapshot of one tenant's credentials.

        Called by ``_invoke`` WITHOUT holding ``_client_lock``: this is where the STS AssumeRole
        network round-trip happens, and it must never block other tenants' cache hits on the shared
        instance. Nothing in here reads or writes the published ``self.*`` cache state; the caller
        publishes the returned client together with exactly these keys under the lock.
        """
        boto_session = None
        if region:
            if access_key and secret_key:
                boto_session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region,
                )
            else:
                boto_session = boto3.Session(region_name=region)
        else:
            boto_session = boto3.Session()

        # If assume role arn is specified, assume the role
        if role_arn:

            from botocore.credentials import RefreshableCredentials
            from botocore.session import get_session

            # Snapshot of the source-account identity for this build. The first AssumeRole and
            # every automatic refresh use this same snapshot, never the published self.* state
            # (which may belong to another tenant).
            refresh_using = functools.partial(
                self._refresh_token,
                access_key=access_key,
                secret_key=secret_key,
                region=region,
                role_arn=role_arn,
            )

            session_credentials = RefreshableCredentials.create_from_metadata(
                metadata=refresh_using(),
                refresh_using=refresh_using,
                method="sts-assume-role"
            )

            session = get_session()
            session._credentials = session_credentials
            session.set_config_variable("region", region)

            boto_session = boto3.Session(botocore_session=session)

        return boto_session.client("sagemaker-runtime")

    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> RerankResult:
        """
        Invoke rerank model

        :param model: model name
        :param credentials: model credentials
        :param query: search query
        :param docs: docs for reranking
        :param score_threshold: score threshold
        :param top_n: top n
        :param user: unique user id
        :return: rerank result
        """
        line = 0
        try:
            if len(docs) == 0:
                return RerankResult(model=model, docs=docs)

            line = 1
            # Read this call's credentials into locals. self.* is only compared against them here;
            # it is not written until the new client has been built successfully (see below).
            new_access_key = credentials.get("aws_access_key_id")
            new_secret_key = credentials.get("aws_secret_access_key")
            new_region = credentials.get("aws_region")
            new_assume_role_arn = credentials.get("assume_role_arn")
            new_endpoint = credentials.get("sagemaker_endpoint")

            with self._client_lock:
                # Short critical section 1 (no I/O): compare this call's keys against the published
                # state and, on a hit, capture the matching client in the same step so a concurrent
                # publish for another tenant cannot swap it underneath us.
                if self.sagemaker_client is not None and \
                    self.access_key == new_access_key and \
                    self.secret_key == new_secret_key and \
                    self.aws_region == new_region and \
                    self.assume_role_arn == new_assume_role_arn and \
                    self.sagemaker_endpoint == new_endpoint:
                    sm_client = self.sagemaker_client
                else:
                    sm_client = None

            if sm_client is None:
                # Cache miss (first call, any credential field changed, or a previous build failed):
                # build a new client OUTSIDE the lock from the immutable locals above. The STS
                # AssumeRole round-trip inside _build_client therefore never blocks other threads'
                # cache hits on this shared instance (dify runs one instance per model type and
                # dispatches requests from a thread pool; the lock is shared class-wide).
                #
                # Invariant: the five cache keys and self.sagemaker_client are only ever written
                # together, under the lock, after the client has been built from exactly those keys.
                # So "all five keys equal" implies "self.sagemaker_client was built from those keys".
                # A failed build raises here, before anything is published, so the previously
                # published (still consistent) state is left untouched: the next call for the failing
                # tenant compares unequal and rebuilds instead of reusing another tenant's client,
                # while other tenants' cache hits are unaffected by the failure.
                sm_client = self._build_client(new_access_key, new_secret_key, new_region, new_assume_role_arn)

                with self._client_lock:
                    # Short critical section 2 (no I/O): atomic publish, keys and client together.
                    self.access_key = new_access_key
                    self.secret_key = new_secret_key
                    self.aws_region = new_region
                    self.assume_role_arn = new_assume_role_arn
                    self.sagemaker_endpoint = new_endpoint
                    self.sagemaker_client = sm_client

            line = 2

            sagemaker_endpoint = credentials.get("sagemaker_endpoint")
            candidate_docs = []

            scores = self._sagemaker_rerank(sm_client, query, docs, sagemaker_endpoint)
            for idx in range(len(scores)):
                candidate_docs.append({"content": docs[idx], "score": scores[idx]})

            sorted(candidate_docs, key=operator.itemgetter("score"), reverse=True)

            line = 3
            rerank_documents = []
            for idx, result in enumerate(candidate_docs):
                rerank_document = RerankDocument(
                    index=idx, text=result.get("content"), score=result.get("score", -100.0)
                )

                if score_threshold is not None:
                    if rerank_document.score >= score_threshold:
                        rerank_documents.append(rerank_document)
                else:
                    rerank_documents.append(rerank_document)

            return RerankResult(model=model, docs=rerank_documents)

        except ClientError as e:
            # Classify AWS-side errors (from STS AssumeRole or the runtime endpoint) so that the
            # _invoke_error_mapping can surface an actionable authorization / rate-limit error.
            logger.exception(f"Failed to invoke rerank model, model: {model}")
            error_code = (getattr(e, "response", None) or {}).get("Error", {}).get("Code", "")
            message = f"Failed to invoke rerank model, model: {model}, error: {str(e)}"
            if error_code in _AUTH_ERROR_CODES:
                raise InvokeAuthorizationError(message) from e
            if error_code in _RATE_LIMIT_ERROR_CODES:
                raise InvokeRateLimitError(message) from e
            raise InvokeError(message) from e
        except Exception as e:
            logger.exception(f"Failed to invoke rerank model, model: {model}")
            raise InvokeError(f"Failed to invoke rerank model, model: {model}, error: {str(e)}")

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            self._invoke(
                model=model,
                credentials=credentials,
                query="What is the capital of the United States?",
                docs=[
                    "Carson City is the capital city of the American state of Nevada. At the 2010 United States "
                    "Census, Carson City had a population of 55,274.",
                    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that "
                    "are a political division controlled by the United States. Its capital is Saipan.",
                ],
                score_threshold=0.8,
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [InvokeConnectionError],
            InvokeServerUnavailableError: [InvokeServerUnavailableError],
            InvokeRateLimitError: [InvokeRateLimitError],
            InvokeAuthorizationError: [InvokeAuthorizationError],
            InvokeBadRequestError: [InvokeBadRequestError, KeyError, ValueError],
        }

    def get_customizable_model_schema(self, model: str, credentials: dict) -> Optional[AIModelEntity]:
        """
        used to define customizable model schema
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.RERANK,
            model_properties={},
            parameter_rules=[],
        )

        return entity
