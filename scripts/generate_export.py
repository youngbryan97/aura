import os
import shutil
import subprocess
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path("/Users/bryan/Desktop/aura")
EXPORT_BASE = Path("/Users/bryan/Downloads")
MAX_FOLDER_FILES = 1000
MAX_CHARS_PER_PART = 4_000_000

# Exclusion patterns (for binary/data/env)
EXCLUDE_DIRS = {
    ".git", ".github", "node_modules", "models", "data", "venv", ".venv",
    "__pycache__", "dist", "build", "archive", "assets", "docker", 
    "experiments", "research", "storage", ".aura_runtime", ".aura_snapshots",
    "test_data", "test_brain", ".pyre", ".pytest_cache", ".ruff_cache", "mujoco_screen.png", "aura_icon.icns", "aura_icon.png"
}

# Essential Infrastructure Extensions (Highly inclusive for logic)
INCLUDE_EXTS = {".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".plist", ".js", ".ts", ".txt", ".TXT", ".log", ".LOG", ".rs", ".cpp", ".h", ".c", ".md", ".MD", ".sql", ".ini"}

# Prioritized directory order
PRIORITY_DIRS = [
    "core", "infrastructure", "interface", "llm", "memory", "optimizer", "security", "skills",
    "integration", "executors", "memory_store", "requirements", "rust_extensions", "systemd",
    "scripts", "tools", "utils", "tests", "docs"
]

def get_target_files():
    """Identifies all infrastructural files for the text parts (everything) 
    and the copy folder (capped at 1000)."""
    
    infra_files_all = []
    seen = set()
    
    # 1. First, include root-level files (important for audit and README)
    root_files = sorted([f for f in SOURCE_DIR.iterdir() if f.is_file()])
    for f in root_files:
        if f.suffix in INCLUDE_EXTS:
            if f.name in {".DS_Store", "package-lock.json"}:
                continue
            if f.name in EXCLUDE_DIRS:
                continue
            if str(f) not in seen:
                infra_files_all.append(f)
                seen.add(str(f))
    
    # 2. Then, get files in the prioritized directories
    for p_dir in PRIORITY_DIRS:
        target = SOURCE_DIR / p_dir
        if not target.exists():
            continue
            
        for root, dirs, filenames in os.walk(target):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            # Sort filenames for deterministic ordering
            filenames.sort()
            
            for f in filenames:
                full_path = Path(root) / f
                ext = full_path.suffix
                if ext in INCLUDE_EXTS:
                    if f in {".DS_Store", "package-lock.json"}:
                        continue
                    if str(full_path) not in seen:
                        infra_files_all.append(full_path)
                        seen.add(str(full_path))
            
    return infra_files_all

def main():
    print("🔍 Scanning for the ENTIRE architecture and infrastructure (High Inclusivity)...")
    all_files = get_target_files()
    num_total = len(all_files)
    print(f"✅ Found {num_total} architecture/infrastructure files.")
    
    # 1. Physical Source Code Copy (Folder: exactly 1000 files maximum)
    copy_dir = EXPORT_BASE / "aura_source_code"
    if copy_dir.exists():
        subprocess.run(["rm", "-rf", str(copy_dir)])
    copy_dir.mkdir(parents=True)
    
    copy_limit = min(MAX_FOLDER_FILES, num_total)
    copy_count = 0
    for f in all_files[:copy_limit]:
        try:
            rel_path = f.relative_to(SOURCE_DIR)
            dest = copy_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            copy_count += 1
        except Exception:
            continue
    print(f"📦 Physical copy complete: {copy_count} files in {copy_dir}.")
    
    # 2. Multi-Part Source parts (ENTIRE infrastructure)
    # The user specifies 5 parts @ 4M chars is expected.
    print(f"📂 Generating source code parts (reaching 5 parts)...")
    current_part = []
    current_size = 0
    part_num = 1
    
    # Clean up previous parts first
    for old_part in EXPORT_BASE.glob("Aura_Source_Part*.txt"):
        old_part.unlink()
        
    for f in all_files:
        try:
            # Basic text-ness check
            if f.stat().st_size > 10_000_000:
                continue
                
            with open(f, "r", encoding="utf-8", errors="ignore") as file_handle:
                content = file_handle.read()
                header = f"\n\n{'='*80}\nFILE: {f.relative_to(SOURCE_DIR)}\n{'='*80}\n\n"
                full_entry = header + content
                
                # Split oversized files
                if len(full_entry) > MAX_CHARS_PER_PART:
                    if current_part:
                        with open(EXPORT_BASE / f"Aura_Source_Part{part_num}.txt", "w") as out:
                            out.write("".join(current_part))
                        print(f"📄 Saved Source Part {part_num} ({current_size} chars).")
                        part_num += 1
                    
                    print(f"⚠️  Splitting large file: {f.name}")
                    for i in range(0, len(full_entry), MAX_CHARS_PER_PART):
                        chunk = full_entry[i:i+MAX_CHARS_PER_PART]
                        if not chunk: continue
                        with open(EXPORT_BASE / f"Aura_Source_Part{part_num}.txt", "w") as out:
                            out.write(chunk)
                        print(f"📄 Saved Source Part {part_num} (Chunk {i//MAX_CHARS_PER_PART} - {len(chunk)} chars).")
                        part_num += 1
                    
                    current_part = []
                    current_size = 0
                elif current_size + len(full_entry) > MAX_CHARS_PER_PART:
                    # Flush current part
                    filename = EXPORT_BASE / f"Aura_Source_Part{part_num}.txt"
                    with open(filename, "w", encoding="utf-8") as out:
                        out.write("".join(current_part))
                    print(f"📄 Saved Source Part {part_num} ({current_size} chars).")
                    
                    # Reset
                    part_num += 1
                    current_part = [full_entry]
                    current_size = len(full_entry)
                else:
                    current_part.append(full_entry)
                    current_size += len(full_entry)
        except Exception:
            continue
            
    if current_part:
        filename = EXPORT_BASE / f"Aura_Source_Part{part_num}.txt"
        with open(filename, "w", encoding="utf-8") as out:
            out.write("".join(current_part))
        print(f"📄 Saved Source Part {part_num} ({current_size} chars).")
    
    print("✨ Audit export successfully completed.")

if __name__ == "__main__":
    main()
