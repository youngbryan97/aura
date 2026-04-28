import os
import shutil
from pathlib import Path

def generate_txt_export():
    root = Path("/Users/bryan/Desktop/aura")
    downloads = Path("/Users/bryan/Downloads")
    
    # Folders considered "Architecture and Infrastructure".  ``research``
    # is included because it owns load-bearing modules (e.g.
    # phi_approximation, causal_emergence) that are referenced from
    # ARCHITECTURE.md and core/.  ``slo`` exposes the SLO contract
    # baseline + measurement harness; ``aura_bench`` carries the
    # capability-delta runner.
    arch_folders = [
        "core",
        "interface",
        "infrastructure",
        "scripts",
        "experiments",
        "tools",
        "skills",
        "research",
        "slo",
        "aura_bench",
    ]
    
    all_files = []
    for folder in arch_folders:
        dir_path = root / folder
        if dir_path.exists():
            for p in dir_path.rglob("*"):
                if p.is_file() and not p.name.startswith(".") and "__pycache__" not in str(p):
                    # Skip large binary files
                    if p.suffix.lower() not in [".py", ".sh", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".yaml", ".yml", ".md", ".json", ".txt"]:
                        continue
                    if p.stat().st_size > 1_000_000: # Skip files > 1MB in the text part
                        continue
                    all_files.append(p)

    # Sort files by importance for the folder copy
    all_files.sort(key=lambda x: (
        0 if "core" in str(x) else 1 if "interface" in str(x) else 2 if "infrastructure" in str(x) else 3,
        str(x)
    ))

    # 1. Generate Folder Copy (Top 1000)
    copy_dir = downloads / "aura_source_copy"
    if copy_dir.exists():
        shutil.rmtree(copy_dir)
    copy_dir.mkdir(parents=True)
    
    for p in all_files[:1000]:
        rel_path = p.relative_to(root)
        target = copy_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    
    print(f"✅ Created folder copy with {len(all_files[:1000])} files in {copy_dir}")

    # 2. Generate Multi-part .txt export
    MAX_CHARS = 4_000_000
    current_part = 1
    current_chars = 0
    current_content = []
    
    part_stats = []

    for p in all_files:
        try:
            rel_path = p.relative_to(root)
            header = f"\n\n{'='*80}\nFILE: {rel_path}\n{'='*80}\n\n"
            content = p.read_text(encoding="utf-8", errors="replace")
            full_entry = header + content
            
            if current_chars + len(full_entry) > MAX_CHARS and current_content:
                # Flush current part
                out_path = downloads / f"aura_source_part_{current_part}.txt"
                final_text = "".join(current_content)
                out_path.write_text(final_text, encoding="utf-8")
                part_stats.append((out_path.name, len(final_text), len(final_text.splitlines())))
                
                current_part += 1
                current_chars = 0
                current_content = []
            
            current_content.append(full_entry)
            current_chars += len(full_entry)
        except Exception as e:
            print(f"Skipping {p}: {e}")

    # Flush last part
    if current_content:
        out_path = downloads / f"aura_source_part_{current_part}.txt"
        final_text = "".join(current_content)
        out_path.write_text(final_text, encoding="utf-8")
        part_stats.append((out_path.name, len(final_text), len(final_text.splitlines())))

    print("\n⏺ Done. Here's what was generated:\n")
    print("  Text files (in ~/Downloads/):")
    print("  ┌" + "─"*24 + "┬" + "─"*13 + "┬" + "─"*9 + "┐")
    print("  │" + "      File".ljust(24) + "│" + "    Size".ljust(13) + "│" + "  Lines".ljust(9) + "│")
    print("  ├" + "─"*24 + "┼" + "─"*13 + "┼" + "─"*9 + "┤")
    for name, size, lines in part_stats:
        size_str = f"~{size/1_000_000:.1f}M chars"
        print(f"  │ {name.ljust(22)} │ {size_str.ljust(11)} │ {str(lines).ljust(7)} │")
    print("  └" + "─"*24 + "┴" + "─"*13 + "┴" + "─"*9 + "┘")

if __name__ == "__main__":
    generate_txt_export()
