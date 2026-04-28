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
    
    # Vendored / generated dirs to exclude even when their files match
    # an architecture extension.  node_modules alone contributes >5000
    # files under interface/static which previously drowned out the
    # actual Aura code in the folder copy.
    EXCLUDE_DIR_SEGMENTS = (
        "node_modules",
        "__pycache__",
        ".next",
        ".turbo",
        ".cache",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    )

    def _excluded(path):
        s = str(path)
        return any(f"/{seg}/" in s or s.endswith(f"/{seg}") for seg in EXCLUDE_DIR_SEGMENTS)

    all_files = []
    for folder in arch_folders:
        dir_path = root / folder
        if dir_path.exists():
            for p in dir_path.rglob("*"):
                if not (p.is_file() and not p.name.startswith(".")):
                    continue
                if _excluded(p):
                    continue
                # Skip large binary files
                if p.suffix.lower() not in [".py", ".sh", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".yaml", ".yml", ".md", ".json", ".txt"]:
                    continue
                if p.stat().st_size > 1_000_000: # Skip files > 1MB in the text part
                    continue
                all_files.append(p)

    # Sort files by importance for the folder copy.  research/ is
    # ranked alongside core because it owns load-bearing modules
    # (phi_approximation, causal_emergence, ...).  slo/ and aura_bench
    # carry the SLO contract and capability-delta harness.
    def _priority(p):
        s = str(p)
        if "/core/" in s or s.startswith("core/"):
            return 0
        if "/interface/" in s or s.startswith("interface/"):
            return 1
        if "/infrastructure/" in s or s.startswith("infrastructure/"):
            return 2
        if "/research/" in s or s.startswith("research/"):
            return 3
        if "/slo/" in s or s.startswith("slo/"):
            return 4
        if "/aura_bench/" in s or s.startswith("aura_bench/"):
            return 5
        return 6

    all_files.sort(key=lambda p: (_priority(p), str(p)))

    # 1. Generate Folder Copy.  Cap raised to 1600 so the whole
    # architecture set fits — the current source has ~1480 files and
    # is growing; the previous 1000 cap was silently dropping research/
    # and slo/ off the end.
    FOLDER_COPY_CAP = 1600
    copy_dir = downloads / "aura_source_copy"
    if copy_dir.exists():
        shutil.rmtree(copy_dir)
    copy_dir.mkdir(parents=True)

    for p in all_files[:FOLDER_COPY_CAP]:
        rel_path = p.relative_to(root)
        target = copy_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)

    print(
        f"✅ Created folder copy with {len(all_files[:FOLDER_COPY_CAP])} files "
        f"in {copy_dir} (cap {FOLDER_COPY_CAP}; total architecture files: "
        f"{len(all_files)})"
    )

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
