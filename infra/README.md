# AWS: a starting template

`template.yaml` deploys SchemaLens serverless, on demand: you pay per document, almost nothing when
idle. It is **a template to adapt**, not a finished platform - see [what you still need to do](#what-you-still-need-to-do).
To run SchemaLens on your own server instead, use Docker (see the main README); nothing here is needed.

```
your system ──► API Gateway (REST) ──► Lambda (container: FastAPI + EasyOCR) ──► your LLM
                 x-api-key, usage plan:        stateless, nothing stored          (OpenAI-compatible or Groq)
                 throttling + daily quota
EventBridge, every 5 min ──► the same Lambda: loads the OCR models (fewer cold starts)
```

- **No database.** API keys, throttling and quotas are handled by API Gateway; the function keeps no
  state and documents are processed in memory only.
- **Optional public demo** (`PublicDemo=true`): adds a Function URL with response streaming
  (live NDJSON progress), called with a demo key (`DemoKey`), and per-address daily counters in
  DynamoDB - only for a public demo page that must stay within a budget. The SchemaLens landing page uses this mode.

## First time on AWS

What the words mean, in one line each:

- **Account** - everything you create belongs to it and is billed to it. Nothing here touches anyone
  else's account, and nothing of yours is in this repository.
- **Region** - where it runs, e.g. `eu-central-1` (Frankfurt). Pick one and stay in it.
- **IAM** - users and permissions. Do not work as the root user: turn on MFA for root and create
  a user in IAM Identity Center for daily work.
- **CloudFormation stack** - the set of resources `template.yaml` describes, created, updated and
  deleted together. **SAM** builds the image and deploys the stack.

Steps:

1. Create the account, turn on MFA for root, create an IAM Identity Center user with
   administrator access.
2. Install the tools: `brew install awscli aws-sam-cli` (and Docker), then sign in:
   `aws configure sso` and `export AWS_PROFILE=<the profile name you chose>`.
3. Deploy (below). The first build and upload take several minutes.
4. Read the outputs: `ApiUrl`, `ApiKeyId` and, with the demo, `DemoUrl`.
5. Update: change the code, `sam build --use-buildkit && sam deploy` again. Remove everything: `sam delete`
   (the stack and all its resources, so its costs stop too).

`sam deploy --guided` saves your answers to `samconfig.toml` next to the template. It is in
`.gitignore`: keep it out of any public repository, and pass secrets (`LlmApiKey`, `DemoKey`,
`IpSalt`) at deploy time rather than writing them down in it.

## Deploy

Needs the AWS CLI, the AWS SAM CLI and Docker.

```bash
cd infra
sam build --use-buildkit        # builds services/api/Dockerfile (target: lambda, arm64)
sam deploy --guided             # pick a region and fill in the parameters below
```

| Parameter | Default | What |
|---|---|---|
| `LlmProfile` | `openai` | `openai` = any OpenAI-compatible endpoint; `groq` = Groq |
| `LlmApiKey` | | key of that endpoint |
| `LlmBaseUrl` | `https://api.openai.com/v1` | endpoint of the `openai` profile (Azure OpenAI, vLLM, OpenRouter...) |
| `LlmModel` | `gpt-4.1-mini` | model name |
| `ThrottleRateLimit` / `ThrottleBurstLimit` | 2 / 5 | requests per second per key |
| `DailyQuota` | 500 | documents per key per day |
| `MaxConcurrency` | 3 | documents processed at the same time (cost guard) |
| `BudgetEmail` / `MonthlyBudgetUsd` | / 5 | monthly cost alert |
| `KillSwitchInvocations` | 600 | more invocations than this in 5 minutes switch the service off (see below) |
| `PublicDemo` | `false` | `true` adds the demo URL and DynamoDB counters |
| `DemoKey` | | public demo only, required there: `schemalens-api keys new --demo`; the page sends it |
| `IpSalt`, `AllowedOrigins`, `DailyDocumentsPerAddress`, `DailyDocumentsTotal` | | public demo only |

## Cost guards

- **Concurrency cap** (`MaxConcurrency`): at most this many documents at once; AWS rejects the rest
  without billing them.
- **Per-key quota and throttling** in API Gateway; **per-address and daily totals** for the demo.
- **Kill switch.** Every call to the public demo URL starts the function and is billed, even one
  rejected with 401. A CloudWatch alarm counts invocations; above `KillSwitchInvocations` in 5
  minutes it sets the function's concurrency to 0 (the service stops, nothing more is billed) and
  emails `BudgetEmail` (confirm the SNS subscription e-mail after the first deploy). Turn it back on:

  ```bash
  aws lambda put-function-concurrency --function-name <ServiceFunctionName output> \
    --reserved-concurrent-executions 3
  # deployed with MaxConcurrency=0:
  aws lambda delete-function-concurrency --function-name <ServiceFunctionName output>
  ```
  A redeploy does not undo it.
- **Budget e-mail** at 80 % of `MonthlyBudgetUsd`. It only warns; the kill switch is what stops.
- **The LLM bill is separate**: use a provider plan with its own hard limit (a free tier without a
  card, or a spending cap).

**New AWS accounts:** the Lambda concurrency quota starts at 10 and AWS keeps 10 unreserved, so
`MaxConcurrency` fails to deploy until you raise it: Service Quotas → AWS Lambda → Concurrent
executions → request 1000 (often approved within minutes; a new account may wait a day or two for a
manual review). Meanwhile deploy with `MaxConcurrency=0`: nothing is reserved and the account
limit (10) caps the function; the kill switch works the same. Redeploy with `MaxConcurrency=3`
once the quota is raised.

Old images stay in the ECR repository SAM creates (about 4 GB each, ~0.10 USD per GB a month):
delete the old ones now and then, or add a lifecycle rule to that repository.

## Call it

```bash
SCHEMALENS_URL=$(aws cloudformation describe-stacks --stack-name schemalens \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)
KEY_ID=$(aws cloudformation describe-stacks --stack-name schemalens \
  --query "Stacks[0].Outputs[?OutputKey=='ApiKeyId'].OutputValue" --output text)
SCHEMALENS_KEY=$(aws apigateway get-api-key --api-key "$KEY_ID" --include-value --query value --output text)

curl -s "$SCHEMALENS_URL/api/v1/extract" -H "x-api-key: $SCHEMALENS_KEY" -F file=@invoice.pdf -F preset=invoice
```

## What you still need to do

The template gives you a working, cost-guarded service. Before real traffic, decide on these
yourself; they depend on your organization:

- **Secrets.** `LlmApiKey` is a stack parameter (`NoEcho`). In production keep it in AWS Secrets
  Manager or SSM Parameter Store and read it at start-up.
- **API keys per client.** The template creates one key and one usage plan. Add keys and plans per
  consumer (and rotate them), or put Cognito / a Lambda authorizer in front if you need identities.
- **Timeouts.** API Gateway (REST) ends an integration after 29 s by default. Hosted LLMs fit easily
  (OCR 5-10 s + LLM a few seconds); long documents or slow models need a higher quota or an
  asynchronous flow.
- **Custom domain and TLS.** An API Gateway custom domain with an ACM certificate (and Route 53).
- **Protection.** AWS WAF on the API (rate rules, IP lists), request size limits suited to you.
- **Monitoring.** CloudWatch alarms on 5xx, latency, throttles and Lambda errors; dashboards; log
  retention to your policy (the template keeps 7 days of metadata-only logs).
- **Network and data residency.** The region, a private API or VPC if required, and the terms of
  the LLM provider the document text is sent to.
- **CI/CD.** `sam deploy` from your pipeline; the image is ~3-4 GB (OCR models baked in).
- **Capacity and cost.** `MaxConcurrency`, memory, the quota and the budget for your volume.

## Costs (an estimate; check the AWS and LLM price lists)

| Item | Idle | With traffic |
|---|---|---|
| Lambda (3 GB, arm64) | warm-up ping, within the free tier | about 5-10 s × 3 GB per document |
| API Gateway (REST) | 0 | per million requests |
| ECR (image of about 3-4 GB) | about 0.10 USD per GB a month | none |
| CloudWatch Logs (7 days) | ≈ 0 | metadata only |
| LLM | 0 | per token at your provider |
| DynamoDB (public demo only) | ≈ 0 | one write per document |

## Privacy

Documents are processed in the function's memory and stored nowhere; logs carry metadata only.
The text read from a document goes to the LLM provider you configure. In the public demo mode,
client addresses are counted only as salted hashes, deleted after two days.
