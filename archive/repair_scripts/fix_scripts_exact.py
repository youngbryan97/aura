import os

replacements = {
    "scripts/aura_collect_flagship_evidence.py": [
        ('json_path.write_text(', 'atomic_write_text(json_path, '),
        ('(out_dir / "flagship_evidence.md").write_text(', 'atomic_write_text((out_dir / "flagship_evidence.md"), ')
    ],
    "scripts/bundle_aura.py": [
        ('output.write_text(', 'atomic_write_text(output, ')
    ],
    "scripts/export_source_bundle.py": [
        ('path.write_text(', 'atomic_write_text(path, ')
    ],
    "scripts/verify_robustness.py": [
        ('temp_file.write_text(', 'atomic_write_text(temp_file, '),
        ('temp_file_2.write_text(', 'atomic_write_text(temp_file_2, '),
        ('temp_file_3.write_text(', 'atomic_write_text(temp_file_3, ')
    ],
    "scripts/aura_morphogenesis_longitudinal_report.py": [
        ('json_path.write_text(', 'atomic_write_text(json_path, '),
        ('(out_dir / "morphogenesis_longitudinal_report.md").write_text(', 'atomic_write_text((out_dir / "morphogenesis_longitudinal_report.md"), ')
    ],
    "scripts/generate_architecture_report.py": [
        ('output_path.write_text(', 'atomic_write_text(output_path, ')
    ],
    "scripts/cognitive_crucible.py": [
        ('RESULTS_PATH.write_text(', 'atomic_write_text(RESULTS_PATH, '),
        ('SUMMARY_PATH.write_text(', 'atomic_write_text(SUMMARY_PATH, ')
    ],
    "scripts/patch_paths.py": [
        ('file_path.write_text(', 'atomic_write_text(file_path, ')
    ],
    "scripts/one_off/live_aura_skill_probe.py": [
        ('ARTIFACT_PATH.write_text(', 'atomic_write_text(ARTIFACT_PATH, ')
    ],
    "scripts/one_off/verify_zenith.py": [
        ('test_data.write_text(', 'atomic_write_text(test_data, ')
    ],
    "scripts/one_off/live_orchestrator_trace_payload.py": [
        ('TRACE_PATH.write_text(', 'atomic_write_text(TRACE_PATH, ')
    ],
    "scripts/aura_task_ownership_codemod.py": [
        ('backup.write_text(', 'atomic_write_text(backup, '),
        ('path.write_text(', 'atomic_write_text(path, ')
    ]
}

for filepath, repls in replacements.items():
    with open(filepath, 'r') as f:
        content = f.read()
    orig = content
    for old, new in repls:
        content = content.replace(old, new)
    
    if content != orig:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                lines.insert(i, 'from core.runtime.atomic_writer import atomic_write_text')
                break
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

import re
# Now fix asyncio in scripts
for root, _, files in os.walk('scripts'):
    for f in files:
        if f.endswith('.py'):
            filepath = os.path.join(root, f)
            with open(filepath, 'r') as f_in:
                content = f_in.read()
            orig = content
            content = re.sub(r'\basyncio\.create_task\(', 'get_task_tracker().create_task(', content)
            content = re.sub(r'\basyncio\.ensure_future\(', 'get_task_tracker().track(', content)
            if content != orig:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        lines.insert(i, 'from core.utils.task_tracker import get_task_tracker')
                        break
                with open(filepath, 'w') as f_out:
                    f_out.write('\n'.join(lines))

