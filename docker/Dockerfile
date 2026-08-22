# Aura Luna Sandbox - Dockerfile
# Provides a secure, network-isolated environment for code execution.

FROM python:3.12-slim

# Security: Run as non-privileged user
RUN groupadd -r aura && useradd -r -g aura aura

WORKDIR /home/aura

# ── Dependencies ─────────────────────────────────────────────────────────
# This is the image untrusted code executes in, so it is the one image whose
# contents must be exactly known. `pip install numpy pandas requests` with no
# bounds made the environment whatever the index served on build day.
COPY requirements/sandbox.txt requirements/sandbox.txt
COPY requirements/lock/code-sandbox.txt requirements/lock/code-sandbox.txt
RUN pip install --no-cache-dir --require-hashes -r requirements/lock/code-sandbox.txt

# Create workspace
RUN mkdir /home/aura/workspace && chown aura:aura /home/aura/workspace

USER aura

# No network entrypoint
CMD ["python3"]
