import os
import shutil
from pathlib import Path

# Configuration
SOURCE_ROOT = "/Users/bryan/.aura/live-source"
DEST_DOWNLOADS = os.path.expanduser("~/Downloads")
DEST_FOLDER_COPY = os.path.join(DEST_DOWNLOADS, "aura_source_copy")
CHAR_LIMIT = 3_999_000 # Strict limit
FILE_LIMIT = 1000

# Inclusion list
INCLUDE_DIRS = [
    "core",
    "infrastructure",
    "llm",
    "native",
    "optimizer",
    "senses",
    "skills",
    "utils",
    "scripts",
    "tools",
    "interface",
]

PRIORITY_MAP = {
    "core": 0, "infrastructure": 1, "llm": 2, "native": 3, "optimizer": 4,
    "senses": 5, "skills": 6, "utils": 7, "scripts": 8, "tools": 9, "interface": 10,
}

EXTENSIONS = {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".js", ".jsx", ".html", ".css", ".md"}
FILENAMES = {"Dockerfile", "Makefile", "aura_main.py", "main_daemon.py", "Aura.entitlements"}

def get_priority(path):
    rel_path = os.path.relpath(path, SOURCE_ROOT)
    parts = rel_path.split(os.sep)
    if parts[0] in PRIORITY_MAP: return PRIORITY_MAP[parts[0]]
    return 20

def collect_files():
    all_files = []
    for dname in INCLUDE_DIRS:
        dir_path = os.path.join(SOURCE_ROOT, dname)
        if not os.path.exists(dir_path): continue
        for root, dirs, files in os.walk(dir_path):
            if any(x in root for x in ["__pycache__", ".venv", "models", "data", "dist", "build"]): continue
            for f in files:
                if os.path.splitext(f)[1] in EXTENSIONS:
                    f_path = os.path.join(root, f)
                    if os.path.splitext(f)[1] == ".json" and os.path.getsize(f_path) > 1_000_000: continue
                    all_files.append(f_path)
    for f in os.listdir(SOURCE_ROOT):
        if f in FILENAMES or (f.endswith(".py") and os.path.isfile(os.path.join(SOURCE_ROOT, f))):
            all_files.append(os.path.join(SOURCE_ROOT, f))
    files = list(set(all_files))
    files.sort(key=lambda x: (get_priority(x), x))
    return files

def export_to_txt(files):
    for f in os.listdir(DEST_DOWNLOADS):
        if f.startswith("aura_source_part_") and f.endswith(".txt"): os.remove(os.path.join(DEST_DOWNLOADS, f))
            
    part_num = 1
    output_parts = []
    current_content = []
    current_size = 0
    current_lines = 0

    def flush_part():
        nonlocal part_num, current_content, current_size, current_lines
        if not current_content: return
        f_out_path = os.path.join(DEST_DOWNLOADS, f"aura_source_part_{part_num}.txt")
        full_text = "".join(current_content)
        with open(f_out_path, 'w', encoding='utf-8') as f: f.write(full_text)
        output_parts.append((f"aura_source_part_{part_num}.txt", len(full_text), full_text.count('\n')))
        part_num += 1
        current_content = []
        current_size = 0
        current_lines = 0

    for f_path in files:
        rel_path = os.path.relpath(f_path, SOURCE_ROOT)
        try:
            with open(f_path, 'r', encoding='utf-8') as f: content = f.read()
        except: continue
        
        header = f"\n\n{'='*80}\nFILE: {rel_path}\n{'='*80}\n"
        entry = header + content
        
        while entry:
            space_left = CHAR_LIMIT - current_size
            if len(entry) <= space_left:
                current_content.append(entry)
                current_size += len(entry)
                current_lines += entry.count('\n')
                entry = ""
            else:
                current_content.append(entry[:space_left])
                entry = entry[space_left:]
                flush_part()
    flush_part()
    return output_parts

def export_to_folder(files):
    if os.path.exists(DEST_FOLDER_COPY): shutil.rmtree(DEST_FOLDER_COPY)
    os.makedirs(DEST_FOLDER_COPY)
    for f_path in files[:FILE_LIMIT]:
        rel_path = os.path.relpath(f_path, SOURCE_ROOT)
        dest_path = os.path.join(DEST_FOLDER_COPY, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(f_path, dest_path)
    return min(len(files), FILE_LIMIT)

if __name__ == "__main__":
    print("🚀 Starting Perfect Aura Source Export...")
    files = collect_files()
    txt_stats = export_to_txt(files)
    folder_count = export_to_folder(files)
    print(f"Total files: {len(files)}, Parts: {len(txt_stats)}")
    for name, size, lines in txt_stats: print(f"{name}: {size} chars, {lines} lines")
