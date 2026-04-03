# Installation Guide

## Prerequisites
- Python 3.9+
- Ollama running locally (`llama3.1:8b` recommended)
- `ffmpeg` (for voice processing, optional)

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Configure Environment** (Optional):
   Create a `.env` file or export variables:
   ```bash
   export AURA_HOST=0.0.0.0
   export AURA_API_TOKEN=your-secret-token
   ```

3. **Launch Aura**:
   ```bash
   python aura_launcher.py
   ```
   or if installed:
   ```bash
   aura
   ```

4. **Access UI**:
   - Dashboard: `http://localhost:8000/static/launcher.html`
   - Telemetry: `http://localhost:8000/static/telemetry.html`

## TroubleShooting

- **Identity Error**: If `identity_base.txt` is missing, the launcher will create a default one.
- **Ollama Connection**: Ensure Ollama is running on port 11434. Check `core/config.py` if different.
- **Permission Denied**: Run `chmod +x aura_launcher.py`.