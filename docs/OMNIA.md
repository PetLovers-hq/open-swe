# PetLovers Omnia integration

This fork adds Omnia direct messages as a first-class Open SWE source for Agent Luna.

## Open SWE service variables

```dotenv
OMNIA_WEBHOOK_SECRET=
OMNIA_CALLBACK_URL=https://omnia.petlovers.com/api/agents/luna/open-swe
OMNIA_CALLBACK_SECRET=
OMNIA_TOOL_URL=https://omnia.petlovers.com/api/agents/luna/open-swe/tool
OMNIA_TOOL_SECRET=
OMNIA_REPO_OWNER=PetLovers-hq
OMNIA_REPO_NAME=Omnia
OMNIA_AGENT_MODEL=openai:gpt-5.6-luna
OMNIA_AGENT_EFFORT=high
```

`OMNIA_WEBHOOK_SECRET` must equal Omnia's `OPEN_SWE_OMNIA_WEBHOOK_SECRET`.
`OMNIA_CALLBACK_SECRET` must equal Omnia's `OPEN_SWE_OMNIA_CALLBACK_SECRET`.
`OMNIA_TOOL_SECRET` must equal Omnia's `OPEN_SWE_OMNIA_TOOL_SECRET`.

The regular Open SWE completion settings are also required:

```dotenv
COMPLETION_WEBHOOK_URL=https://<open-swe-host>/webhooks/run-complete
RUN_COMPLETE_WEBHOOK_SECRET=
ALLOWED_GITHUB_ORGS=PetLovers-hq
```

## Contract

- `POST /webhooks/omnia` verifies an HMAC over the exact request body.
- An Omnia DM channel maps to one deterministic durable LangGraph thread.
- The default model is `openai:gpt-5.6-luna` at `high` effort.
- `omnia_dm_reply` returns natural-language progress and results to the originating DM.
- `omnia_agent_action` brokers task reads, task creation, and approved merges through Omnia;
  business and merge credentials never enter the coding sandbox.
- The run-completion webhook independently reports terminal success, error, and timeout.
- Omnia journals callback body hashes, so webhook retries cannot duplicate messages.
