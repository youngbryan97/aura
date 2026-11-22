# Install Aura from source

Use an Apple Silicon Mac and Python 3.12. The 27B configuration used on the
project's Mac targets 64 GB of unified memory. Storage and memory requirements
depend on the model files and the lanes you enable. [Hardware profiles](HARDWARE_PROFILES.md)

The checkout contains the application source. You need compatible local
Apple MLX model files to use chat. The persona-tuned 27B checkpoint and
the offline Wikipedia index are separate from the source release.
To repeat the published substrate test, use the [replication kit](tools/replicate/README.md),
which uses a stand-in for the language model.

## Set up the Python environment

```bash
git clone https://github.com/youngbryan97/aura.git
cd aura
make setup-prod PYTHON=python3.12
source .venv/bin/activate
```

`make setup-prod` installs the core runtime requirements and stops on an
installation error. Use `make setup PYTHON=python3.12` if you also need the
test and lint dependencies. Optional dependencies are listed by feature in
[requirements](requirements/).

## Select the local model

Set `AURA_LLM__MLX_MODEL_PATH` to the directory containing your compatible
MLX checkpoint. An installed system can also resolve a promoted checkpoint
from its active model record. A different checkpoint needs its own claims
about behavior, resource use, and compatible steering or adapters.
[Model card](MODEL_CARD.md) · [Model registry](core/brain/llm/model_registry.py)

| Environment variable | Source default | Role |
|---|---|---|
| `AURA_MODEL` | `Aura-Cortex` | Logical name of the foreground model. |
| `AURA_BRAINSTEM_MODEL` | `Ternary-Bonsai-2-27B-mlx-2bit` | Background model, when available and admitted. |

The environment or project-root `.env` can provide settings. Pydantic uses
double underscores for nested settings. See the [operator guide](docs/OPERATOR_GUIDE.md)
for configuration and diagnostics.

## Start the app

In the activated environment, with your model path set:

```bash
python aura_main.py doctor
python aura_main.py --desktop
```

The desktop mode opens the app. API modes listen on `127.0.0.1:8000` by
default. Use `--headless` for the local API without the desktop window or
`--gui-window` to attach to an existing runtime. `--host` and `--port` set the
API bind address. Docker is for headless services and does not supply the
Apple MLX desktop model runtime. [Docker definition](docker-compose.yml)

If the app fails to load a model, check the resolved checkpoint path and
the machine's available memory. An existing Aura process can already own
the model and its port. [Model loading runbook](docs/runbooks/model-fails-to-load.md)

## Next steps

- [Chat, voice, and memory](docs/USER_GUIDE.md)
- [Diagnostics, backup, and updates](docs/OPERATOR_GUIDE.md)
- [Tests](TESTING.md)
