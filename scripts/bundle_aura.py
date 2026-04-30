from core.runtime.atomic_writer import atomic_write_text
import os
from pathlib import Path

def bundle_codebase(root_dir, output_file, max_size_mb=25):
    root = Path(root_dir).resolve()
    output = Path(output_file).resolve()
    
    # Files to include
    extensions = {'.py', '.md', '.html', '.css', '.js', '.command', '.sh'}
    whitelist_dirs = {
        'core', 'interface', 'senses', 'memory', 'embodiment', 'security', 
        'scripts', 'brain', 'docs', 'tests', 'cradle', 'skills'
    }
    # Files to include
    extensions = {'.py', '.md', '.html', '.css', '.js', '.command', '.sh'}
    
    bundle_content = []
    total_size = 0
    file_count = 0
    
    print(f"📦 Bundling Aura from {root}...")
    
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
            
        try:
            rel_path = path.relative_to(root)
            # Only include if first part of rel_path is in whitelist_dirs 
            # OR if it's a top-level file
            if len(rel_path.parts) > 1 and rel_path.parts[0] not in whitelist_dirs:
                continue
            # Also skip common junk files
            if rel_path.name.startswith(('.', 'aura_consolidated', 'aura_source', 'aura_sovereign')):
                continue
        except ValueError:
            continue

        if path.suffix in extensions:
            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                
                header = f"\n{'='*80}\nFILE: {rel_path}\n{'='*80}\n"
                file_bundle = header + content
                
                bundle_content.append(file_bundle)
                total_size += len(file_bundle.encode('utf-8'))
                file_count += 1
                
                if total_size > max_size_mb * 1024 * 1024:
                    print(f"⚠️ Warning: Bundle size exceeds {max_size_mb}MB. Stopping.")
                    break
            except Exception as e:
                print(f"❌ Failed to read {path}: {e}")
                
    atomic_write_text(output, "\n".join(bundle_content), encoding='utf-8')
    print(f"✅ Bundle created: {output} ({total_size / 1024 / 1024:.2f} MB)")
    print(f"📄 Total files: {file_count}")

if __name__ == "__main__":
    # Ensure we use the correct absolute path
    base_dir = Path(__file__).resolve().parent.parent
    output_path = base_dir.parent / "aura_consolidated_source.txt"
    bundle_codebase(base_dir, output_path)
