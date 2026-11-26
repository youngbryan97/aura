import os
import shutil
from pathlib import Path


def generate_txt_export():
    root = Path(os.environ.get("AURA_SOURCE_DIR", Path(__file__).resolve().parents[1])).expanduser().resolve()
    downloads = Path(os.environ.get("AURA_EXPORT_DIR", Path.home() / "Downloads")).expanduser().resolve()
    max_parts = int(os.environ.get("AURA_EXPORT_MAX_PARTS", "10"))
    copy_limit = int(os.environ.get("AURA_EXPORT_COPY_LIMIT", "1000"))
    
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
    exclude_dir_segments = (
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
        return any(f"/{seg}/" in s or s.endswith(f"/{seg}") for seg in exclude_dir_segments)

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

    # 1. Generate Folder Copy.  Default cap matches the user's requested
    # review bundle shape; raise AURA_EXPORT_COPY_LIMIT when a complete
    # lower-priority docs/config copy is more important than the cap.
    folder_copy_cap = copy_limit
    copy_dir = downloads / "aura_source_copy"
    if copy_dir.exists():
        shutil.rmtree(copy_dir)
    copy_dir.mkdir(parents=True)

    for p in all_files[:folder_copy_cap]:
        rel_path = p.relative_to(root)
        target = copy_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)

    print(
        f"Created folder copy with {len(all_files[:folder_copy_cap])} files "
        f"in {copy_dir} (cap {folder_copy_cap}; total architecture files: "
        f"{len(all_files)})"
    )

    # 2. Generate Multi-part .txt export
    max_chars = 4_000_000
    current_part = 1
    current_chars = 0
    current_content = []
    
    part_stats = []
    text_files_included = 0
    truncated = False

    for old_part in downloads.glob("aura_source_part_*.txt"):
        old_part.unlink()

    for p in all_files:
        try:
            rel_path = p.relative_to(root)
            header = f"\n\n{'='*80}\nFILE: {rel_path}\n{'='*80}\n\n"
            content = p.read_text(encoding="utf-8", errors="replace")
            full_entry = header + content
            
            if current_chars + len(full_entry) > max_chars and current_content:
                if current_part >= max_parts:
                    truncated = True
                    break
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
            text_files_included += 1
        except (OSError, UnicodeError) as e:
            print(f"Skipping {p}: {e}")

    # Flush last part
    if current_content:
        out_path = downloads / f"aura_source_part_{current_part}.txt"
        final_text = "".join(current_content)
        out_path.write_text(final_text, encoding="utf-8")
        part_stats.append((out_path.name, len(final_text), len(final_text.splitlines())))

    print("\nDone. Here's what was generated:\n")
    print("  Text files (in ~/Downloads/):")
    print("  ┌" + "─"*24 + "┬" + "─"*13 + "┬" + "─"*9 + "┐")
    print("  │" + "      File".ljust(24) + "│" + "    Size".ljust(13) + "│" + "  Lines".ljust(9) + "│")
    print("  ├" + "─"*24 + "┼" + "─"*13 + "┼" + "─"*9 + "┤")
    for name, size, lines in part_stats:
        size_str = f"~{size/1_000_000:.1f}M chars"
        print(f"  │ {name.ljust(22)} │ {size_str.ljust(11)} │ {str(lines).ljust(7)} │")
    print("  └" + "─"*24 + "┴" + "─"*13 + "┴" + "─"*9 + "┘")
    summary = (
        "Aura source export summary\n"
        f"Source root: {root}\n"
        f"Text parts: {len(part_stats)} / max {max_parts}\n"
        f"Character cap per part: {max_chars}\n"
        f"Text files included: {text_files_included} / {len(all_files)}\n"
        f"Text export truncated by max parts: {truncated}\n"
        f"Folder copy: {copy_dir}\n"
        f"Folder files copied: {len(all_files[:folder_copy_cap])} / cap {folder_copy_cap}\n"
        "Priority: core, interface, infrastructure, research, slo, aura_bench, then remaining scripts/tools/skills.\n"
    )
    (downloads / "aura_source_export_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

if __name__ == "__main__":
    generate_txt_export()
