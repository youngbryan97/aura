# Aura Luna Sandbox - Dockerfile
# Provides a secure, network-isolated environment for code execution.

FROM python:3.12-slim

# Security: Run as non-privileged user
RUN groupadd -r aura && useradd -r -g aura aura

WORKDIR /home/aura

# Pre-install common safe libraries if needed
RUN pip install --no-cache-dir numpy pandas requests

# Create workspace
RUN mkdir /home/aura/workspace && chown aura:aura /home/aura/workspace

USER aura

# No network entrypoint
CMD ["python3"]
