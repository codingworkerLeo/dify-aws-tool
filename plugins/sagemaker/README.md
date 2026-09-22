## Amazon Sagemaker

**Author:** aws  
**Type:** Model Provider



## Overview | 概述

The [Amazon Sagemaker](https://aws.amazon.com/sagemaker/) is a fully managed service that brings together a broad set of tools to enable high-performance, low-cost ML for any use case. With SageMaker AI, you can build, train and deploy ML models at scale using tools like notebooks, debuggers, profilers, pipelines, MLOps, and more – all in one integrated development environment (IDE).

[Amazon Sagemaker](https://aws.amazon.com/sagemaker/) 是一项完全托管的服务，它汇集了广泛的工具集，为任何用例提供高性能、低成本的机器学习能力。通过 SageMaker AI，您可以使用笔记本、调试器、性能分析器、管道、MLOps 等工具在一个集成开发环境 (IDE) 中大规模构建、训练和部署机器学习模型。



## Configure | 配置

After installing the plugin, configure the Sagemaker endpoint url within the Model Provider settings. Obtain your endpoint url from [here](https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin). Once saved, you can begin using Sagemaker to build your AI agents and agentic workflows.

安装插件后，在模型提供商设置中配置 Sagemaker 端点 URL。您可以从[这里](https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin)获取端点 URL。保存后，您就可以开始使用 Sagemaker 构建 AI 代理和代理工作流。

![](./_assets/sagemaker_model.PNG)

You could add model through the `settings -> model provider -> Sagemaker` page.

您可以通过 `设置 -> 模型提供商 -> Sagemaker` 页面添加模型。

![](./_assets/sagemaker_config.PNG)

## Cross-Account Access with AssumeRole | 使用 AssumeRole 进行跨账户访问

The SageMaker plugin supports cross-account access using AWS AssumeRole functionality. This allows you to access SageMaker endpoints deployed in different AWS accounts while maintaining security best practices.

SageMaker 插件支持使用 AWS AssumeRole 功能进行跨账户访问。这允许您访问部署在不同 AWS 账户中的 SageMaker 端点，同时保持安全最佳实践。

**Supported model types**: AssumeRole cross-account access is available for **LLM**, **Text Embedding** and **Rerank** models. All three model types share the same `assume_role_arn` field in the provider's model credential schema, so the configuration steps below apply to each of them in exactly the same way. The plugin uses STS temporary credentials with automatic refresh (`RefreshableCredentials`), so no manual credential rotation is needed for long-running deployments.

**支持的模型类型**：AssumeRole 跨账户访问适用于 **LLM**、**Text Embedding（文本嵌入）** 和 **Rerank（重排序）** 三种模型。这三种模型类型在模型提供商的凭证配置（model credential schema）中共用同一个 `assume_role_arn` 字段，因此下文的配置步骤对三者完全一致。插件使用 STS 临时凭证并自动续签（`RefreshableCredentials`），长期运行的部署无需手动轮换凭证。

### When to Use AssumeRole | 何时使用 AssumeRole

- **Multi-account architecture**: When your SageMaker endpoints are in a different AWS account than your Dify deployment
- **Enhanced security**: Use temporary credentials instead of long-term access keys
- **Enterprise environments**: Meet compliance requirements for cross-account resource access
- **DevOps workflows**: Separate development, testing, and production environments across accounts

- **多账户架构**：当您的 SageMaker 端点与 Dify 部署位于不同的 AWS 账户中时
- **增强安全性**：使用临时凭证而不是长期访问密钥
- **企业环境**：满足跨账户资源访问的合规要求
- **DevOps 工作流**：在不同账户中分离开发、测试和生产环境

### Setup Instructions | 设置说明

#### 1. Create IAM Role in Target Account | 在目标账户中创建 IAM 角色

In the AWS account where your SageMaker endpoint is deployed:

在部署 SageMaker 端点的 AWS 账户中：

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint",
                "sagemaker:DescribeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:*:*:endpoint/*"
        }
    ]
}
```

#### 2. Configure Trust Relationship | 配置信任关系

Set up the trust policy to allow the source account to assume this role:

设置信任策略以允许源账户承担此角色：

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::SOURCE-ACCOUNT-ID:root"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "sts:ExternalId": "optional-external-id"
                }
            }
        }
    ]
}
```

#### 3. Grant AssumeRole Permission in Source Account | 在源账户中授予 AssumeRole 权限

In your Dify deployment account, ensure the IAM user/role has permission to assume the target role:

在您的 Dify 部署账户中，确保 IAM 用户/角色有权限承担目标角色：

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Resource": "arn:aws:iam::TARGET-ACCOUNT-ID:role/SageMakerCrossAccountRole"
        }
    ]
}
```

#### 4. Configure in Dify | 在 Dify 中配置

When adding a SageMaker model in Dify, fill in the **Assume Role ARN** field:

在 Dify 中添加 SageMaker 模型时，填写 **跨账户角色ARN** 字段：

```
arn:aws:iam::TARGET-ACCOUNT-ID:role/SageMakerCrossAccountRole
```

The **Assume Role ARN** field is defined once in the provider's `model_credential_schema` and is shown for **LLM**, **Text Embedding** and **Rerank** models alike. Pick the corresponding **Model Type** when adding the model, then fill in the same set of fields:

**跨账户角色ARN** 字段在模型提供商的 `model_credential_schema` 中只定义一次，对 **LLM**、**Text Embedding** 和 **Rerank** 三种模型类型同样显示。添加模型时选择对应的 **模型类型**，然后填写同一组字段即可：

**Text Embedding example | Text Embedding 示例**

| Field / 字段 | Value / 取值 |
|-------|-------|
| Model Type / 模型类型 | `Text Embedding` |
| Model Name / 模型名称 | `bge-m3-embedding` |
| SageMaker Endpoint / SageMaker 端点 | `bge-m3-embedding-endpoint` |
| AWS Region / AWS 地区 | `us-east-1` |
| Access Key / Secret Access Key | Optional, source account credentials / 可选，源账户凭证 |
| Assume Role ARN / 跨账户角色ARN | `arn:aws:iam::TARGET-ACCOUNT-ID:role/SageMakerCrossAccountRole` |

**Rerank example | Rerank 示例**

| Field / 字段 | Value / 取值 |
|-------|-------|
| Model Type / 模型类型 | `Rerank` |
| Model Name / 模型名称 | `bge-reranker-v2-m3` |
| SageMaker Endpoint / SageMaker 端点 | `bge-reranker-v2-m3-endpoint` |
| AWS Region / AWS 地区 | `us-east-1` |
| Access Key / Secret Access Key | Optional, source account credentials / 可选，源账户凭证 |
| Assume Role ARN / 跨账户角色ARN | `arn:aws:iam::TARGET-ACCOUNT-ID:role/SageMakerCrossAccountRole` |

For **Text Embedding** and **Rerank** models, the plugin first builds a session from the source account credentials (explicit Access Key / Secret Access Key, or the runtime environment's default credential chain), then calls `sts:AssumeRole` on the target role and invokes the endpoint with the temporary credentials. For **LLM** models, the session used to call `sts:AssumeRole` is currently built from the configured AWS Region and the runtime environment's default credential chain only; the explicit Access Key / Secret Access Key are not used for that AssumeRole call. If **Assume Role ARN** is left empty, the plugin behaves exactly as before and invokes the endpoint directly with the source account credentials.

对于 **Text Embedding** 和 **Rerank** 两种模型类型，插件会先用源账户凭证（显式的 Access Key / Secret Access Key，或运行环境的默认凭证链）建立会话，再对目标角色调用 `sts:AssumeRole`，并使用临时凭证调用端点。对于 **LLM** 模型类型，目前调用 `sts:AssumeRole` 的会话仅基于配置的 AWS 地区和运行环境的默认凭证链建立，显式填写的 Access Key / Secret Access Key 不会用于该 AssumeRole 调用。如果 **跨账户角色ARN** 留空，插件行为与之前完全一致，直接使用源账户凭证调用端点。

### Configuration Options | 配置选项

| Field | Required | Description |
|-------|----------|-------------|
| **Access Key** | Optional | Source account credentials (can use IAM role instead) |
| **Secret Access Key** | Optional | Source account credentials (can use IAM role instead) |
| **Assume Role ARN** | Optional | Target account role ARN for cross-account access |
| **AWS Region** | Required | Region where the SageMaker endpoint is deployed |
| **SageMaker Endpoint** | Required | The endpoint name to invoke |

| 字段 | 必填 | 描述 |
|------|------|------|
| **Access Key** | 可选 | 源账户凭证（可以使用 IAM 角色代替） |
| **Secret Access Key** | 可选 | 源账户凭证（可以使用 IAM 角色代替） |
| **跨账户角色ARN** | 可选 | 用于跨账户访问的目标账户角色 ARN |
| **AWS 地区** | 必填 | 部署 SageMaker 端点的地区 |
| **SageMaker 端点** | 必填 | 要调用的端点名称 |

### Security Best Practices | 安全最佳实践

- **Principle of least privilege**: Grant only the minimum permissions required
- **Use external IDs**: Add external ID conditions for additional security
- **Monitor access**: Use CloudTrail to monitor cross-account access
- **Rotate credentials**: Temporary credentials are automatically rotated
- **Network security**: Consider VPC endpoints for private connectivity

- **最小权限原则**：仅授予所需的最小权限
- **使用外部 ID**：添加外部 ID 条件以增强安全性
- **监控访问**：使用 CloudTrail 监控跨账户访问
- **轮换凭证**：临时凭证会自动轮换
- **网络安全**：考虑使用 VPC 端点进行私有连接

### Troubleshooting | 故障排除

**Common Issues:**

- **Access Denied**: Check IAM permissions and trust relationships
- **Invalid ARN**: Verify the role ARN format and account ID
- **Region Mismatch**: Ensure the region matches your SageMaker endpoint
- **Endpoint Not Found**: Verify the endpoint name and deployment status

**常见问题：**

- **访问被拒绝**：检查 IAM 权限和信任关系
- **无效的 ARN**：验证角色 ARN 格式和账户 ID
- **地区不匹配**：确保地区与您的 SageMaker 端点匹配
- **端点未找到**：验证端点名称和部署状态

## Examples & Feedback | 示例 & 反馈

For more detailed information, please refer to [aws-sample/dify-aws-tool](https://github.com/aws-samples/dify-aws-tool/), which contains multiple workflows for reference.

If you have issues that need feedback, feel free to raise questions or look for answers in the [Issue](https://github.com/aws-samples/dify-aws-tool/issues) section.

更多详细信息可以参考 [aws-sample/dify-aws-tool](https://github.com/aws-samples/dify-aws-tool/)，其中包含多个 workflow 供参考。

如果存在问题需要反馈，欢迎到 [Issue](https://github.com/aws-samples/dify-aws-tool/issues) 去提出问题或者寻找答案。
