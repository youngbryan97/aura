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

def build_gemini_bundle():
    root_dir = Path("/Users/bryan/.gemini/antigravity/scratch/autonomy_engine")
    output_file = Path.home() / "Desktop" / "Aura_Gemini_Audit_Full.txt"
    
    # Dirs we definitely want to include fully
    include_dirs = {'core', 'interface', 'skills', 'autonomic'}
    exclude_dirs = {'.git', 'venv', '__pycache__', 'data', 'artifacts', 'brain', '.gemini', 'node_modules', 'dist', 'build', 'logs', 'backups', 'scipy', 'matplotlib', 'pandas'}

    bundle_content = "AURA ARCHITECTURE AUDIT (GEMINI 3.1 PRO)\n"
    bundle_content += "=================================================\n"
    bundle_content += "Constraints: Under 4,000,000 characters (approx 1M tokens) up to 100MB.\n"
    bundle_content += "Goal: Comprehensive code layout for full systems analysis.\n"
    bundle_content += "=================================================\n\n"
    
    char_limit = 3900000 # 3.9M to be safe under 4,000,000
    current_chars = len(bundle_content)
    processed = set()
    
    # Add files fully as long as we constrain to the char limit
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if not file.endswith((".py", ".md", ".json", ".txt", ".yml", ".sh")): continue
            if file in {"package-lock.json"}: continue
            
            filepath = Path(root) / file
            rel = filepath.relative_to(root_dir)
            top_level = rel.parts[0] if rel.parts else ""
            
            # Allow top-level files like aura_main.py, requirements.txt, or files in included dirs
            if top_level not in include_dirs and len(rel.parts) > 1:
                if top_level not in {'.github', 'scripts', 'tests'}:
                    continue
                
            if str(filepath.resolve()) in processed:
                continue
            
            # Try to read and add content
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                heading = f"\n--- FULL FILE: {rel} ---\n"
                addition = heading + content + "\n"
                
                if current_chars + len(addition) < char_limit:
                    bundle_content += addition
                    current_chars += len(addition)
                    processed.add(str(filepath.resolve()))
                else:
                    # Not enough room for full file, just summarize if python
                    if file.endswith(".py"):
                        summary = get_file_summary(filepath)
                        addition = f"\n--- SUMMARY ONLY: {rel} ---\n{summary}\n"
                        if current_chars + len(addition) < char_limit:
                            bundle_content += addition
                            current_chars += len(addition)
                            processed.add(str(filepath.resolve()))
                    else:
                        addition = f"\n--- SKIPPED FILE (Over Limit): {rel} ---\n"
                        if current_chars + len(addition) < char_limit:
                            bundle_content += addition
                            current_chars += len(addition)
            except Exception as e:
                # ignore files we can't read
                pass

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(bundle_content)
        
    print(f"✅ Generated {output_file.name} successfully.")
    print(f"📊 Final size: {len(bundle_content):,} characters / {char_limit:,} max limit.")

if __name__ == '__main__':
    build_gemini_bundle()
