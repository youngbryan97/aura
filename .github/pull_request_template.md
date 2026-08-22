## What changed

<!-- One paragraph. What is different after this lands, and for whom. -->

## Why

<!-- The behaviour that was wrong, or the capability that was missing.
     Link the incident, the failing gate, or the issue. -->

## How it was checked

<!-- Name the gate or the test, not the intent. "make smoke" and the file
     that now fails without this change. A claim with no check is a claim. -->

- [ ] `make compile lint smoke` pass locally
- [ ] the behaviour has a test that fails without this change
- [ ] any new claim about the system is registered in
      `core/organism/model_validation.py` with the test that validates it
- [ ] no ratchet baseline in `config/` grew

## Risk

<!-- What breaks if this is wrong, and how it would be noticed. -->

## Review

<!-- CODEOWNERS assigns the reviewer. If the change touches security, the
     runtime foundation, persistent state, model execution or the release
     contract, say here what an attacker or an operator would try first. -->
