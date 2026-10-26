# Runbook: External Egress Privacy Incident

<!-- reader-navigation:start -->
> **In plain English:** Security describes what information may cross a boundary and what untrusted input is allowed to influence.
>
> **Why it matters:** An assistant sees documents, websites and tool results that can contain instructions it should not treat as permission from the user.
>
> **In everyday use:** A web page telling the assistant to reveal a private file must remain page content rather than becoming an instruction to send the file.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#security) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#security) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
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
