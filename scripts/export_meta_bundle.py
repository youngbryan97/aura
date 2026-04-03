import os
import ast
from pathlib import Path

def get_file_summary(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree) or ""
        
        classes = []
        functions = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
                
        summary = f"Docstring: {docstring.split(chr(10))[0] if docstring else 'None'}\n"
        if classes: summary += f"  Classes: {', '.join(classes)}\n"
        if functions: summary += f"  Functions: {', '.join(functions)}\n"
        return summary.strip()
    except Exception:
        return "Not summarizable (parse error or non-python)"

def build_meta_bundle():
    root_dir = Path("/Users/bryan/.gemini/antigravity/scratch/autonomy_engine")
    output_file = Path.home() / "Desktop" / "Aura_Meta_Audit.txt"
    
    priority_files = [
        "core/orchestrator.py",
        "core/config.py",
        "core/orchestrator_boot.py",
        "interface/server.py",
        "core/mycelium.py"
    ]
    
    include_dirs = {'core', 'interface', 'skills', 'autonomic'}
    exclude_dirs = {'.git', 'venv', '__pycache__', 'data', 'artifacts', 'brain', '.gemini', 'node_modules', 'dist', 'build', 'logs', 'backups', 'scipy', 'matplotlib', 'pandas'}

    bundle_content = "AURA ARCHITECTURE AUDIT (META LLAMA 4 MAVERICK)\n"
    bundle_content += "=================================================\n"
    bundle_content += "Constraints: Under 100,000 characters. Most critical files included fully.\n"
    bundle_content += "=================================================\n\n"
    
    char_limit = 95000
    current_chars = len(bundle_content)
    processed = set()
    
    # 1. Add Priority Files fully
    for rel_path in priority_files:
        filepath = root_dir / rel_path
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                heading = f"\n--- FULL FILE: {rel_path} ---\n"
                addition = heading + content + "\n"
                
                if current_chars + len(addition) < (char_limit - 15000): # Reserve 15k chars for summaries
                    bundle_content += addition
                    current_chars += len(addition)
                    processed.add(str(filepath.resolve()))
                else:
                    processed.add(str(filepath.resolve()))
                    addition = f"\n--- SUMMARY ONLY: {rel_path} ---\n{get_file_summary(filepath)}\n"
                    bundle_content += addition
                    current_chars += len(addition)

    # 2. Add Summaries for the rest
    bundle_content += "\n=================================================\n"
    bundle_content += "SYSTEM ARCHITECTURE HIGH-LEVEL SUMMARIES (Truncated for Context limits)\n"
    bundle_content += "=================================================\n"
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if not file.endswith(".py"): continue
            
            filepath = Path(root) / file
            rel = filepath.relative_to(root_dir)
            top_level = rel.parts[0] if rel.parts else ""
            
            if top_level not in include_dirs:
                continue
                
            if str(filepath.resolve()) in processed:
                continue
                
            summary = get_file_summary(filepath)
            addition = f"\n📄 File: {rel}\n  {summary}\n"
            
            if current_chars + len(addition) < char_limit:
                 bundle_content += addition
                 current_chars += len(addition)
            else:
                 break

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(bundle_content)
        
    print(f"✅ Generated {output_file.name} successfully.")
    print(f"📊 Final size: {len(bundle_content)} characters.")

if __name__ == '__main__':
    build_meta_bundle()
