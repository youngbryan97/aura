# Runbook: External Egress Privacy Incident

<!-- reader-navigation:start -->
> Aura can read files and webpages that contain instructions. Permission checks have to keep those instructions separate from what the user authorized.
>
> A mistaken permission can expose information or let an app change files. If a webpage asks Aura to send a private file, the text stays page content. An actual send needs the user’s authority.
>
> [Plain-language guide](../PLAIN_ENGLISH_GUIDE.md#security) · [Terms and examples](../VARIABLES_EXPLAINED.md#security) · [Evidence and related work](../DEMOS_AND_RELATED_WORK.md#runtime)
> [All docs](../DOCUMENTATION_INDEX.md) · [Glossary](../GLOSSARY.md) · [Worked formulas](../FORMULAS.md)
<!-- reader-navigation:end -->

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
