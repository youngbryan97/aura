# Runbook: External Egress Privacy Incident

## Trigger

- A governed web/API action sends sensitive content unexpectedly
- Egress privacy inspection or redaction fails
- An external service reports a data incident

## Diagnosis

1. Stop the affected external action lane and identify its source:
   ```bash
   grep "egress_privacy\|NetworkGateway" ~/.aura/logs/*.log | tail -50
   ```

2. Review governed network receipts:
   ```bash
   grep "network_gateway" ~/.aura/logs/*.log | tail -50
   ```

3. Identify the destination, source component, request digest and whether the
   body was inspected, transformed or refused. Never paste the sensitive body
   into another external service during diagnosis.

## Response

1. **Quarantine the affected capability or destination immediately.** Do not
   disable the process-wide egress gateway; that would create an ungoverned
   bypass rather than containment.

2. **Audit sent payloads** using local receipts and destination records.

3. **Assess privacy impact** and rotate any exposed credential.

4. **Repair the owning adapter** and prove the fix through
   `core.security.egress_privacy` plus `NetworkGateway`; do not add a direct
   HTTP client as a workaround.

## Prevention

- Remote model inference is not part of Aura's runtime.
- External network actions pass through `NetworkGateway`.
- Structured bodies are inspected and secrets are redacted or refused.
- Egress decisions carry local audit evidence without logging secret values.
