import functools
import itertools
import json
import logging
import threading
import time
from typing import Any, Optional

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from dify_plugin.entities.model import (
    AIModelEntity,
    EmbeddingInputType,
    FetchFrom,
    I18nObject,
    ModelPropertyKey,
    ModelType,
    PriceType,
)
from dify_plugin.entities.model.text_embedding import EmbeddingUsage, TextEmbeddingResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.text_embedding_model import TextEmbeddingModel

BATCH_SIZE = 20
CONTEXT_SIZE = 8192

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


def batch_generator(generator, batch_size):
    while True:
        batch = list(itertools.islice(generator, batch_size))
        if not batch:
            break
        yield batch


class SageMakerEmbeddingModel(TextEmbeddingModel):
    """
    Model class for Cohere text embedding model.
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

    def _sagemaker_embedding(self, sm_client, endpoint_name, content_list: list[str]):
        response_model = sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps({"inputs": content_list, "parameters": {}, "is_query": False, "instruction": ""}),
            ContentType="application/json",
        )
        json_str = response_model["Body"].read().decode("utf8")
        json_obj = json.loads(json_str)
        embeddings = json_obj["embeddings"]
        return embeddings

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
            "RoleSessionName": f"dify-sagemaker-embedding-{int(time.time())}"
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
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Invoke text embedding model

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :param user: unique user id
        :param input_type: input type
        :return: embeddings result
        """
        # get model properties
        try:
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

            line = 3
            truncated_texts = [item[:CONTEXT_SIZE] for item in texts]

            batches = batch_generator((text for text in truncated_texts), batch_size=BATCH_SIZE)
            all_embeddings = []

            line = 4
            for batch in batches:
                embeddings = self._sagemaker_embedding(sm_client, sagemaker_endpoint, batch)
                all_embeddings.extend(embeddings)

            line = 5
            # calc usage
            usage = self._calc_response_usage(
                model=model,
                credentials=credentials,
                tokens=0,  # It's not SAAS API, usage is meaningless
            )
            line = 6

            return TextEmbeddingResult(embeddings=all_embeddings, usage=usage, model=model)

        except ClientError as e:
            # Classify AWS-side errors (from STS AssumeRole or the runtime endpoint) so that the
            # _invoke_error_mapping can surface an actionable authorization / rate-limit error.
            logger.exception(f"Failed to invoke text embedding model, model: {model}, line: {line}")
            error_code = (getattr(e, "response", None) or {}).get("Error", {}).get("Code", "")
            if error_code in _AUTH_ERROR_CODES:
                raise InvokeAuthorizationError(str(e)) from e
            if error_code in _RATE_LIMIT_ERROR_CODES:
                raise InvokeRateLimitError(str(e)) from e
            raise InvokeError(str(e)) from e
        except Exception as e:
            logger.exception(f"Failed to invoke text embedding model, model: {model}, line: {line}")
            raise InvokeError(str(e))

    def get_num_tokens(self, model: str, credentials: dict, texts: list[str]) -> list[int]:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :return:
        """
        return [0] * len(texts)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            print("validate_credentials ok....")
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _calc_response_usage(self, model: str, credentials: dict, tokens: int) -> EmbeddingUsage:
        """
        Calculate response usage

        :param model: model name
        :param credentials: model credentials
        :param tokens: input tokens
        :return: usage
        """
        # get input price info
        input_price_info = self.get_price(
            model=model, credentials=credentials, price_type=PriceType.INPUT, tokens=tokens
        )

        # transform usage
        usage = EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=time.perf_counter() - self.started_at,
        )

        return usage

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [InvokeConnectionError],
            InvokeServerUnavailableError: [InvokeServerUnavailableError],
            InvokeRateLimitError: [InvokeRateLimitError],
            InvokeAuthorizationError: [InvokeAuthorizationError],
            InvokeBadRequestError: [KeyError],
        }

    def get_customizable_model_schema(self, model: str, credentials: dict) -> Optional[AIModelEntity]:
        """
        used to define customizable model schema
        """

        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.TEXT_EMBEDDING,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: CONTEXT_SIZE,
                ModelPropertyKey.MAX_CHUNKS: BATCH_SIZE,
            },
            parameter_rules=[],
        )

        return entity
