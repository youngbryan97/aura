import os
import ast
import time
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("/Users/bryan/.gemini/antigravity/scratch/autonomy_engine")
DESKTOP_PATH = Path("/Users/bryan/Desktop")

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", 
    "venv", ".venv", "node_modules", "data", "models", "dist", "logs", "build", "artifacts", "brain", ".gemini"
}
EXCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".pyc", ".db", ".sqlite", ".zip", ".tar.gz", ".icns", ".obj"}

def get_file_summary(filepath):
    """Generate a high-level technical summary of a Python file using AST."""
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
        
        summary = [f"### TECHNICAL SUMMARY: {filepath.name}", docstring.split('\n')[0], ""]
        if classes:
            summary.append("Classes:")
            summary.extend(classes)
        if functions:
            summary.append("Functions:")
            summary.extend(functions)
        
        return "\n".join(summary)
    except Exception as e:
        return f"### SUMMARY UNAVAILABLE: {filepath.name} (Error: {e})"

def generate_snapshot(name, char_limit, filename):
    print(f"Generating {filename} for {name} (Limit: {char_limit:,} chars)...")
    buffer = []
    current_chars = 0
    
    header = f"================================================================================\n"
    header += f"AURA SYSTEM ARCHITECTURE SNAPSHOT - FOR {name.upper()}\n"
    header += f"Generated on: {time.ctime()}\n"
    header += f"Target Constraint: {char_limit:,} characters\n"
    header += f"================================================================================\n\n"
    
    buffer.append(header)
    current_chars += len(header)
    
    # Prioritize certain directories
    priority_dirs = ['core', 'interface', 'skills', 'autonomic', 'infrastructure']
    
    all_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
            all_files.append(Path(root) / file)
            
    # Sort files: priority dirs first, then by path
    def sort_key(p):
        rel = p.relative_to(PROJECT_ROOT)
        parts = rel.parts
        if not parts: return (1, str(rel))
        top = parts[0]
        if top in priority_dirs:
            return (0, priority_dirs.index(top), str(rel))
        return (1, str(rel))

    all_files.sort(key=sort_key)
    
    included_count = 0
    summarized_count = 0
    
    for file_path in all_files:
        rel_path = file_path.relative_to(PROJECT_ROOT)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_header = f"\nFILE: {rel_path}\n" + "-"*40 + "\n"
            footer = "\n" + "="*80 + "\n"
            entry_len = len(file_header) + len(content) + len(footer)
            
            if current_chars + entry_len < char_limit:
                buffer.append(file_header)
                buffer.append(content)
                buffer.append(footer)
                current_chars += entry_len
                included_count += 1
            else:
                # Add summary instead
                summary = get_file_summary(file_path)
                summary_entry = f"\n--- SUMMARY ONLY (Over Limit): {rel_path} ---\n{summary}\n" + "="*80 + "\n"
                if current_chars + len(summary_entry) < char_limit:
                    buffer.append(summary_entry)
                    current_chars += len(summary_entry)
                    summarized_count += 1
        except Exception as e:
            err_msg = f"\nERROR READING {rel_path}: {e}\n"
            if current_chars + len(err_msg) < char_limit:
                buffer.append(err_msg)
                current_chars += len(err_msg)

    output_path = DESKTOP_PATH / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("".join(buffer))
        
    print(f"✅ Created {filename}: {current_chars:,} chars, {included_count} full files, {summarized_count} summaries.\n")

if __name__ == "__main__":
    # Claude: ~0.8 MB = 800,000 chars
    generate_snapshot("Claude", 800_000, "Aura_Audit_Claude.txt")
    
    # Gemini: ~4.0 MB = 4,000,000 chars
    generate_snapshot("Gemini", 4_000_000, "Aura_Audit_Gemini.txt")
    
    # Grok: ~8.0 MB = 8,000,000 chars
    generate_snapshot("Grok", 8_000_000, "Aura_Audit_Grok.txt")
