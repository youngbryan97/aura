import os
import shutil
from pathlib import Path

# Config
SOURCE_DIR = "/Users/bryan/.aura/live-source"
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
TXT_OUT_PREFIX = os.path.join(DOWNLOADS_DIR, "aura_source_part_")
COPY_OUT_DIR = os.path.join(DOWNLOADS_DIR, "aura_source_copy")
MAX_CHARS_PER_FILE = 4_000_000
MAX_FILES_COPY = 1000

# Exclude patterns
EXCLUDE_DIRS = {".git", "venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".aura", "coverage"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz", ".db", ".sqlite", ".sqlite3"}

# Define architecture and infrastructure directories
ARCH_DIRS = {"core", "infrastructure", "interface", "skills", "tools", "scripts", "proof_kernel", "aura_bench"}

def is_valid_file(path, rel_path):
    # Check if inside excluded dir relative to SOURCE_DIR
    parts = rel_path.parts
    if any(p in EXCLUDE_DIRS for p in parts):
        return False
    # Check extension
    if path.suffix.lower() in EXCLUDE_EXTS:
        return False
    # Hide dotfiles except maybe .env example? We'll ignore hidden files.
    if path.name.startswith(".") and path.name not in {".gitignore", ".dockerignore", ".env.example"}:
        return False
    return True

def get_all_files():
    all_files = []
    base_path = Path(SOURCE_DIR)
    for root, dirs, files in os.walk(base_path):
        # Mutate dirs in place to skip excluded
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(base_path)
            if is_valid_file(full_path, rel_path):
                all_files.append(full_path)
    return all_files

def rank_files(files):
    # Rank files by importance
    # 1. core/ *.py
    # 2. infrastructure/ *.py
    # 3. root *.py, Dockerfile, Makefile
    # 4. interface/ *.py, *.js, *.ts, *.html
    # 5. skills/ *.py
    # 6. others
    ranked = []
    for f in files:
        rel_path = f.relative_to(SOURCE_DIR)
        score = 100
        
        parts = rel_path.parts
        if not parts:
            continue
            
        top_level = parts[0]
        if top_level == "core":
            score = 10
            if f.suffix == ".py":
                score = 5
        elif top_level == "infrastructure":
            score = 20
        elif len(parts) == 1 and f.suffix == ".py":
            score = 30
        elif len(parts) == 1 and f.name in {"Dockerfile", "Makefile"}:
            score = 35
        elif top_level == "interface":
            score = 40
        elif top_level == "skills":
            score = 50
        elif top_level == "tools":
            score = 60
        elif top_level == "scripts":
            score = 70
        else:
            score = 90
            
        ranked.append((score, str(rel_path), f))
        
    # Sort by score ascending, then by path
    ranked.sort()
    return [item[2] for item in ranked]

def is_arch_file(f):
    rel = f.relative_to(SOURCE_DIR)
    parts = rel.parts
    if not parts:
        return False
    if parts[0] in ARCH_DIRS:
        return True
    if len(parts) == 1 and f.suffix == ".py":
        return True
    if len(parts) == 1 and f.name in {"Dockerfile", "Makefile", "docker-compose.yml"}:
        return True
    return False

def generate_txt_files(files):
    part_num = 1
    current_chars = 0
    current_content = []
    
    # Filter only arch files for the text dump
    arch_files = [f for f in files if is_arch_file(f)]
    
    file_info = []
    
    for f in arch_files:
        try:
            rel_path = f.relative_to(SOURCE_DIR)
            with open(f, 'r', encoding='utf-8') as file_obj:
                content = file_obj.read()
            
            header = f"\n\n{'='*80}\nFILE: {rel_path}\n{'='*80}\n\n"
            file_str = header + content
            file_len = len(file_str)
            
            if current_chars + file_len > MAX_CHARS_PER_FILE and current_content:
                # Write current part
                out_path = f"{TXT_OUT_PREFIX}{part_num}.txt"
                with open(out_path, 'w', encoding='utf-8') as out_f:
                    out_f.write("".join(current_content))
                
                # Record stats
                file_info.append({
                    "name": f"aura_source_part_{part_num}.txt",
                    "chars": current_chars,
                    "lines": sum(c.count('\n') for c in current_content)
                })
                
                part_num += 1
                current_chars = 0
                current_content = []
            
            current_content.append(file_str)
            current_chars += file_len
            
        except UnicodeDecodeError:
            # Skip binary or non-utf8 files
            pass
        except Exception as e:
            pass

    # Write the last part
    if current_content:
        out_path = f"{TXT_OUT_PREFIX}{part_num}.txt"
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.write("".join(current_content))
            
        file_info.append({
            "name": f"aura_source_part_{part_num}.txt",
            "chars": current_chars,
            "lines": sum(c.count('\n') for c in current_content)
        })

    return file_info, len(arch_files)

def copy_files(files):
    if os.path.exists(COPY_OUT_DIR):
        get_task_tracker().create_task(get_storage_gateway().delete_tree(COPY_OUT_DIR, cause='copy_files'))
    os.makedirs(COPY_OUT_DIR, exist_ok=True)
    
    copied = 0
    for f in files:
        if copied >= MAX_FILES_COPY:
            break
        try:
            # Try to read to ensure it's a text file or valid file
            if f.is_file():
                rel_path = f.relative_to(SOURCE_DIR)
                dest = Path(COPY_OUT_DIR) / rel_path
                get_task_tracker().create_task(get_storage_gateway().create_dir(dest.parent, cause='copy_files'))
                shutil.copy2(f, dest)
                copied += 1
        except Exception:
            pass
    return copied

def main():
    print("Gathering files...")
    all_files = get_all_files()
    
    print(f"Total files found: {len(all_files)}")
    ranked_files = rank_files(all_files)
    
    print("Generating TXT exports...")
    txt_info, arch_count = generate_txt_files(ranked_files)
    
    print("Copying prioritized files to folder...")
    copied_count = copy_files(ranked_files)
    
    print("\n\n=== EXPORT COMPLETE ===")
    print("Text files (in ~/Downloads/):")
    for info in txt_info:
        print(f"  {info['name']}: {info['chars']} chars, {info['lines']} lines")
    
    print(f"\nTotal architecture files exported: {arch_count}")
    print(f"Folder copy (in ~/Downloads/aura_source_copy/): {copied_count} files.")

if __name__ == "__main__":
    main()
