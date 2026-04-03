import os
import ast
import time
import shutil
from pathlib import Path

# --- Configuration ---
SOURCE_ROOT = Path("/Users/bryan/Desktop/aura")
OUTPUT_DIR = Path("/Users/bryan/Desktop")
CHAR_LIMIT = 4_000_000
LITERAL_COPY_LIMIT = 1000

EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules", "data", "models", 
    "dist", "logs", "build", "artifacts", ".mypy_cache", ".pytest_cache", 
    ".ruff_cache", ".idea", ".vscode", "aura/data", "site-packages", "venv"
}

EXCLUDE_EXTS = {
    ".pyc", ".db", ".safetensors", ".png", ".jpg", ".jpeg", ".gif", 
    ".pdf", ".zip", ".tar", ".gz", ".exe", ".icns", ".obj", ".sqlite", ".bin"
}

PRIORITY_DIRS = ['core', 'interface', 'infrastructure', 'llm', 'memory', 'optimizer', 'security', 'skills']

def get_file_summary(filepath):
    """Generate a technical summary using AST."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree) or "No docstring provided."
        
        classes = []
        functions = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(f"  - Class: {node.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(f"  - Function: {node.name}")
        
        summary = [f"### ARCHITECTURAL SUMMARY: {filepath.name}", docstring.split('\n')[0], ""]
        if classes:
            summary.append("Classes:")
            summary.extend(classes)
        if functions:
            summary.append("Functions:")
            summary.extend(functions)
        
        return "\n".join(summary)
    except Exception as e:
        return f"### SUMMARY UNAVAILABLE: {filepath.name} (Error: {e})"

class MultiPartWriter:
    def __init__(self, base_name):
        self.base_name = base_name
        self.part_num = 1
        self.current_chars = 0
        self.buffer = []

    def _flush(self):
        if not self.buffer: return
        filename = f"{self.base_name}_Part{self.part_num}.txt"
        output_path = OUTPUT_DIR / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("".join(self.buffer))
        print(f"✅ Created {filename}: {self.current_chars:,} chars.")
        self.buffer = []
        self.current_chars = 0
        self.part_num += 1

    def add(self, content):
        if self.current_chars + len(content) > CHAR_LIMIT:
            self._flush()
        
        if len(content) > CHAR_LIMIT:
            # If for some reason a single file is > 4M, we have to chunk it
            # This is rare but possible for massive files
            for i in range(0, len(content), CHAR_LIMIT - 100):
                chunk = content[i:i + CHAR_LIMIT - 100]
                self.add(f"\n--- [CONTINUED IN NEXT CHUNK] ---\n{chunk}\n")
        else:
            self.buffer.append(content)
            self.current_chars += len(content)

    def close(self):
        self._flush()

def generate_exports():
    print(f"🚀 Starting Comprehensive Aura Export...")
    
    # 1. Collect and Rank Files
    all_files = []
    for root, dirs, files in os.walk(SOURCE_ROOT):
        # In-place directory filtering
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            path = Path(root) / file
            if path.suffix in EXCLUDE_EXTS: continue
            if not path.exists(): continue
            all_files.append(path)

    # Ranking logic: Priority Dirs first, then depth, then alphabetical
    def sort_key(p):
        rel = p.relative_to(SOURCE_ROOT)
        parts = rel.parts
        if not parts: return (0, 0, str(rel))  # Root files like aura_main.py
        top = parts[0]
        if top in PRIORITY_DIRS:
            return (0, PRIORITY_DIRS.index(top), str(rel))
        return (1, 0, str(rel))

    all_files.sort(key=sort_key)
    
    # LIMIT: Only process the top 2000 files for snapshots to focus on core infrastructure
    # 57k files is mostly data/logs/etc that shouldn't be in the audit.
    SNAPSHOT_LIMIT = 2000
    snapshot_files = all_files[:SNAPSHOT_LIMIT]
    
    # 2. Literal Copy (Top 1000)
    tree_dir = OUTPUT_DIR / "aura_source_tree"
    if tree_dir.exists():
        shutil.rmtree(tree_dir)
    tree_dir.mkdir(parents=True)
    
    print(f"📦 Creating literal copy in {tree_dir}...")
    copy_count = 0
    for file_path in all_files[:LITERAL_COPY_LIMIT]:
        rel_path = file_path.relative_to(SOURCE_ROOT)
        dest_path = tree_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)
        copy_count += 1
    print(f"✅ Copied {copy_count} files to aura_source_tree.")

    # 3. Architecture Export (Multi-part)
    print(f"🧬 Generating Architecture Snapshots (for top {len(snapshot_files)} files)...")
    arch_writer = MultiPartWriter("Aura_Architecture")
    
    header = f"================================================================================\n"
    header += f"AURA ARCHITECTURAL OVERVIEW - SYSTEM SNAPSHOT\n"
    header += f"Generated on: {time.ctime()}\n"
    header += f"Target: Core Infrastructure (Top {len(snapshot_files)} files)\n"
    header += f"================================================================================\n\n"
    arch_writer.add(header)

    for i, file_path in enumerate(snapshot_files):
        if i % 100 == 0: print(f"  ... processing architecture: {i}/{len(snapshot_files)}")
        rel_path = file_path.relative_to(SOURCE_ROOT)
        
        # Only do AST for Python files
        if file_path.suffix == ".py":
            summary = get_file_summary(file_path)
        else:
            summary = f"### FILE SUMMARY: {file_path.name} (Non-Python File)"
            
        entry = f"## FILE: {rel_path}\n{summary}\n" + "="*80 + "\n"
        arch_writer.add(entry)
    
    arch_writer.close()

    # 4. Source Code Export (Multi-part)
    print(f"📂 Generating Source Code Snapshots (for top {len(snapshot_files)} files)...")
    source_writer = MultiPartWriter("Aura_SourceCode")
    source_writer.add(header.replace("ARCHITECTURAL OVERVIEW", "FULL SOURCE CODE"))

    for i, file_path in enumerate(snapshot_files):
        if i % 100 == 0: print(f"  ... processing source: {i}/{len(snapshot_files)}")
        rel_path = file_path.relative_to(SOURCE_ROOT)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            entry = f"\n## FILE: {rel_path}\n" + "-"*40 + "\n"
            entry += f"```python\n# Path: {rel_path}\n{content}\n```\n"
            entry += "="*80 + "\n"
            source_writer.add(entry)
        except Exception as e:
            source_writer.add(f"\n[ERROR READING {rel_path}: {e}]\n")

    source_writer.close()
    print(f"✨ All exports completed successfully.")

if __name__ == "__main__":
    generate_exports()
