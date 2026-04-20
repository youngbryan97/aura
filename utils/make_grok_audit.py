import os
from pathlib import Path

def generate_grok_audit():
    # Target size: 48 MB (approx 50,331,648 bytes)
    target_bytes = 48 * 1024 * 1024
    output_path = Path.home() / "Desktop" / "Aura_Grok_Audit_Full.txt"
    
    project_root = Path(__file__).resolve().parents[1]
    
    print(f"Bundling source for Grok Audit into {output_path}...")
    
    content = []
    content.append("="*80)
    content.append("AURA LUNA: ENTERPRISE AUDIT BUNDLE (v2026.4.20-ZENITH)")
    content.append("AESTHETIC: HIGH FIDELITY | FOCUS: 100% SYNTHESIS")
    content.append("="*80 + "\n")
    
    # 1. Collect all Python source files
    py_files = sorted(list(project_root.rglob("*.py")))
    py_files = [f for f in py_files if "venv" not in str(f) and ".git" not in str(f) and "__pycache__" not in str(f)]
    
    total_source_chars = 0
    for f in py_files:
        try:
            rel_path = f.relative_to(project_root)
            with open(f, 'r', encoding='utf-8') as src:
                code = src.read()
                content.append(f"FILE: {rel_path}")
                content.append("-" * len(str(rel_path)))
                content.append(code)
                content.append("\n" + "#" * 40 + "\n")
                total_source_chars += len(code)
        except Exception as e:
            content.append(f"ERROR READING {f}: {e}")
            
    # Join the initial content
    bundle_str = "\n".join(content)
    current_bytes = len(bundle_str.encode('utf-8'))
    
    # 2. If under target, add high-fidelity duplicates/repeats with "Analysis Views"
    # To reach 48MB, we might need a lot of padding.
    # Grok says it can synthesize 100%, so we'll give it the code multiple times
    # under different "Instructional Framings" to test its reasoning over massive context.
    
    iterations = 0
    while current_bytes < target_bytes:
        iterations += 1
        padding = f"\n\n{'='*80}\n"
        padding += f"ARCHITECTURAL AUDIT VIEW #{iterations} - DEEP SYNTHESIS BUFFER\n"
        padding += f"This section repeats the source for cross-contextual reasoning test.\n"
        padding += f"{'='*80}\n\n"
        
        # Add the whole bundle again for stress testing synthesis
        bundle_str += padding + bundle_str[:target_bytes - current_bytes] # Trim to fit perfectly
        current_bytes = len(bundle_str.encode('utf-8'))
        
        if iterations > 50: # Safety break
            break

    # Final trim to be exactly 48 MB or just under
    final_output = bundle_str.encode('utf-8')[:target_bytes]
    
    with open(output_path, 'wb') as f:
        f.write(final_output)
        
    print(f"Success! Generated {len(final_output)} bytes to {output_path}")

if __name__ == "__main__":
    generate_grok_audit()
