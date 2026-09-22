import json
import logging
import operator
import time
from typing import Any, Optional

import boto3  # type: ignore

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

    def _sagemaker_rerank(self, query_input: str, docs: list[str], rerank_endpoint: str):
        inputs = [query_input] * len(docs)
        response_model = self.sagemaker_client.invoke_endpoint(
            EndpointName=rerank_endpoint,
            Body=json.dumps({"inputs": inputs, "docs": docs}),
            ContentType="application/json",
        )
        json_str = response_model["Body"].read().decode("utf8")
        json_obj = json.loads(json_str)
        scores = json_obj["scores"]
        return scores if isinstance(scores, list) else [scores]

    def _refresh_token(self):
        " Refresh tokens by calling assume_role again "
        params = {
            "RoleArn": self.assume_role_arn,
            "DurationSeconds": 3600,
            "RoleSessionName": f"dify-sagemaker-rerank-{int(time.time())}"
        }

        boto_session = boto3.Session(region_name=self.aws_region)
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
            if self.access_key != credentials.get("aws_access_key_id") or \
                self.secret_key != credentials.get("aws_secret_access_key") or \
                self.aws_region != credentials.get("aws_region") or \
                self.assume_role_arn != credentials.get("assume_role_arn") or \
                self.sagemaker_endpoint != credentials.get("sagemaker_endpoint"):

                # Any credential field changed (or first call): rebuild the client
                self.access_key = credentials.get("aws_access_key_id")
                self.secret_key = credentials.get("aws_secret_access_key")
                self.aws_region = credentials.get("aws_region")
                self.assume_role_arn = credentials.get("assume_role_arn")
                self.sagemaker_endpoint = credentials.get("sagemaker_endpoint")

                boto_session = None
                if self.aws_region:
                    if self.access_key and self.secret_key:
                        boto_session = boto3.Session(
                            aws_access_key_id=self.access_key,
                            aws_secret_access_key=self.secret_key,
                            region_name=self.aws_region,
                        )
                    else:
                        boto_session = boto3.Session(region_name=self.aws_region)
                else:
                    boto_session = boto3.Session()

                # If assume role arn is specified, assume the role
                if self.assume_role_arn:

                    from botocore.credentials import RefreshableCredentials
                    from botocore.session import get_session

                    session_credentials = RefreshableCredentials.create_from_metadata(
                        metadata=self._refresh_token(),
                        refresh_using=self._refresh_token,
                        method="sts-assume-role"
                    )

                    session = get_session()
                    session._credentials = session_credentials
                    session.set_config_variable("region", self.aws_region)

                    boto_session = boto3.Session(botocore_session=session)

                self.sagemaker_client = boto_session.client("sagemaker-runtime")

            line = 2

            sagemaker_endpoint = credentials.get("sagemaker_endpoint")
            candidate_docs = []

            scores = self._sagemaker_rerank(query, docs, sagemaker_endpoint)
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
