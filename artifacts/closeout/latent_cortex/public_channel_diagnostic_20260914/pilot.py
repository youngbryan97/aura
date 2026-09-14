"""One development task compares raw-text stopping with public-channel stopping."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from core.brain.llm.public_channel_decode import decode_public_greedy
from core.brain.llm.unified_recurrent_transfer_decode import decode_base_greedy_tokens
from core.runtime.atomic_writer import atomic_write_bytes_if_absent
from core.runtime.model_lane_control import standalone_model_lane
from tools.run_semantic_neural_decode_canary import _complete, _grade, _prompt_tokens, _task_cohort

ROOT = Path.cwd()
OUT = Path('/tmp/aura-g06-public-20260913/pilot.json')
assert not OUT.exists()
MANIFEST = Path('/Users/bryan/.aura/live-source/training/fused-model/active.json')
raw_manifest = MANIFEST.read_bytes()
MODEL = Path(json.loads(raw_manifest)['active_model_path'])
task = _task_cohort(('coding',), 2, seed=2026091306, surface_profile='canonical')[0]
started = time.monotonic()
with standalone_model_lane(owner_id='g06-public-channel-pilot', model_path=str(MODEL),
                          purpose='evaluation', preemptible=False,
                          require_exclusive=True, allow_owner_eviction=False):
    from mlx_lm import load
    model, tokenizer = load(str(MODEL))
    tokens = _prompt_tokens(tokenizer, task.prompt)
    print(json.dumps({'event': 'loaded', 'prompt_tokens': len(tokens), 'task_id': task.task_id}), flush=True)
    rows = []
    legacy, stopped, latency = decode_base_greedy_tokens(
        model, tokens, eos_token_id=tokenizer.eos_token_id, max_tokens=384,
        completion_check=lambda values: _complete(tokenizer, values),
    )
    legacy_text = tokenizer.decode(list(legacy), skip_special_tokens=True)
    correct, parsed = _grade(task, legacy_text)
    rows.append({'policy': 'legacy_raw_text', 'max_tokens': 384, 'correct': correct, 'parsed': parsed,
                 'generated_tokens': len(legacy), 'stopped': stopped, 'latency_ms': latency,
                 'response_sha256': hashlib.sha256(legacy_text.encode()).hexdigest()})
    print(json.dumps({'event': 'legacy_complete', **rows[-1]}), flush=True)
    def progress(n):
        if n % 256 == 0:
            print(json.dumps({'event': 'public_decode_progress', 'tokens': n}), flush=True)
    from core.brain.llm.latent_cortex.answer_contract import contract_decode_disposition, ContractDecodeDisposition
    result = decode_public_greedy(
        model, tokenizer, tokens, max_tokens=4096, progress=progress,
        completion_check=lambda text: contract_decode_disposition(text) in {
            ContractDecodeDisposition.COMPLETE, ContractDecodeDisposition.INVALID},
    )
    correct, parsed = _grade(task, result.text)
    rows.append({**result.receipt(), 'max_tokens': 4096, 'correct': correct, 'parsed': parsed,
                 'public_response': result.text})
    print(json.dumps({'event': 'public_complete', 'correct': correct, 'parsed': parsed,
                      'generated_tokens': result.generated_tokens, 'stop_reason': result.stop_reason}), flush=True)
payload = {'schema': 'aura.g06.public_channel_pilot.v1', 'scope': 'exposed_single_task_development_diagnostic',
           'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'source_sha256s': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
               'core/brain/llm/public_channel_decode.py', 'core/brain/llm/unified_recurrent_transfer_decode.py',
               'core/brain/llm/chat_format.py', 'tools/run_semantic_neural_decode_canary.py')},
           'model_path': str(MODEL), 'resident_manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
           'task_id': task.task_id, 'seed': 2026091306, 'rows': rows,
           'elapsed_seconds': time.monotonic() - started, 'serving_authority': False,
           'limitations': ['single task', 'budget and channel changes combined',
                           'bare fused model, runtime steering not installed', 'not a gain or qualification claim']}
assert atomic_write_bytes_if_absent(OUT, (json.dumps(payload, sort_keys=True) + '\n').encode('ascii'), mode=0o400)
print(json.dumps({'event': 'complete', 'output': str(OUT)}), flush=True)
