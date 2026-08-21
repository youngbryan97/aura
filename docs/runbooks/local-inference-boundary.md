# Runbook: Local Inference Boundary

## Trigger

- A review asks whether a prompt can reach a model outside this machine
- A request context arrives carrying `allow_cloud_fallback`
- The router logs a fallback to a lane the operator does not recognise
- An incident response has to state where inference happened

## What the boundary is

There is no remote inference path. The router falls back only between local
lanes, named in `core/brain/llm/model_registry.py`:

| Constant | Lane |
| --- | --- |
| `PRIMARY_ENDPOINT` | `Cortex` |
| `DEEP_ENDPOINT` | `Solver` |
| `BRAINSTEM_ENDPOINT` | `Brainstem` |
| `FALLBACK_ENDPOINT` | `Reflex` |

`HealthAwareLLMRouter._fallback_endpoint_names` chooses among those four and
nothing else. It takes `_allow_cloud_fallback` and does not read it; the
leading underscore records that.

The flag is pinned before the router sees it. `validate_request_context` in
`core/brain/request_contract.py` coerces `allow_cloud_fallback` to `False`
whatever the caller supplied, and the field is not in `POLICY_FIELDS`, so no
policy can raise it either. `tests/test_request_contract.py` asserts both.

## Diagnosis

1. Confirm the flag is still pinned:
   ```bash
   .venv/bin/python -m pytest tests/test_request_contract.py -q
   ```

2. List the lanes the router actually chose this session:
   ```bash
   grep "endpoint\|lane" ~/.aura/logs/*.log | tail -50
   ```
   Every name should be one of Cortex, Solver, Brainstem, or Reflex.

3. Check that nothing outbound carried a prompt:
   ```bash
   grep "network_gateway\|egress_privacy" ~/.aura/logs/*.log | tail -50
   ```

## Response

1. **An unrecognised lane name means a provider was registered outside the
   registry.** Find its `register_endpoint` caller and remove it. Do not add
   the name to the registry to make the log tidy.

2. **A prompt in an outbound body is an egress incident, not a routing one.**
   Follow [external-egress.md](external-egress.md).

3. **A caller that sets `allow_cloud_fallback=True` is not a breach** — the
   contract drops it — but it is a caller written against a capability that
   does not exist. Fix the call site.

## Prevention

- Keep `allow_cloud_fallback` out of `POLICY_FIELDS`. A field policy can set
  is a field an operator can turn on.
- New lanes go in `core/brain/llm/model_registry.py`. A lane the registry does
  not name cannot be selected as a fallback.
- `core/capabilities/web_interlocutor.py` is the one governed path that speaks
  to an external AI, and it does it through the user's visible browser under a
  host allowlist and a per-run turn budget. It is not inference and it is not
  a fallback.
