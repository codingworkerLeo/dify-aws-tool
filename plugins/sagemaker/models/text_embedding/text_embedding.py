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

# botocore ClientError codes that mean the request was throttled.
_RATE_LIMIT_ERROR_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "RequestLimitExceeded",
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

    # Guards compare -> build -> publish -> read of the six cached attributes above, so that two
    # threads sharing one model instance can never pair one tenant's client with another's endpoint.
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
                if self.sagemaker_client is None or \
                    self.access_key != new_access_key or \
                    self.secret_key != new_secret_key or \
                    self.aws_region != new_region or \
                    self.assume_role_arn != new_assume_role_arn or \
                    self.sagemaker_endpoint != new_endpoint:

                    # Any credential field changed (or first call, or a previous build failed):
                    # rebuild the client from the local values.
                    #
                    # Invariant: the five cache keys and self.sagemaker_client are always published
                    # together, and only after the new client has been built. So "all five keys
                    # equal" implies "self.sagemaker_client was built from exactly those keys".
                    # If anything below raises, all six attributes are reset to None before
                    # re-raising, so the next call is guaranteed to take this rebuild branch
                    # (self.sagemaker_client is None) instead of reusing a client that was built
                    # for a different tenant.
                    try:
                        boto_session = None
                        if new_region:
                            if new_access_key and new_secret_key:
                                boto_session = boto3.Session(
                                    aws_access_key_id=new_access_key,
                                    aws_secret_access_key=new_secret_key,
                                    region_name=new_region,
                                )
                            else:
                                boto_session = boto3.Session(region_name=new_region)
                        else:
                            boto_session = boto3.Session()

                        # If assume role arn is specified, assume the role
                        if new_assume_role_arn:

                            from botocore.credentials import RefreshableCredentials
                            from botocore.session import get_session

                            # Snapshot of the source-account identity for this build. The first
                            # AssumeRole and every automatic refresh use this same snapshot, never
                            # the published self.* state (which may belong to another tenant).
                            refresh_using = functools.partial(
                                self._refresh_token,
                                access_key=new_access_key,
                                secret_key=new_secret_key,
                                region=new_region,
                                role_arn=new_assume_role_arn,
                            )

                            session_credentials = RefreshableCredentials.create_from_metadata(
                                metadata=refresh_using(),
                                refresh_using=refresh_using,
                                method="sts-assume-role"
                            )

                            session = get_session()
                            session._credentials = session_credentials
                            session.set_config_variable("region", new_region)

                            boto_session = boto3.Session(botocore_session=session)

                        new_client = boto_session.client("sagemaker-runtime")
                    except Exception:
                        # Build failed: invalidate the whole cache so the next call must rebuild.
                        self.sagemaker_client = None
                        self.access_key = None
                        self.secret_key = None
                        self.aws_region = None
                        self.assume_role_arn = None
                        self.sagemaker_endpoint = None
                        raise

                    # Atomic publish: keys and client become visible together.
                    self.access_key = new_access_key
                    self.secret_key = new_secret_key
                    self.aws_region = new_region
                    self.assume_role_arn = new_assume_role_arn
                    self.sagemaker_endpoint = new_endpoint
                    self.sagemaker_client = new_client

                # Capture the client matching this call's credentials while still holding the lock,
                # so a concurrent rebuild for another tenant cannot swap it underneath us.
                sm_client = self.sagemaker_client

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
