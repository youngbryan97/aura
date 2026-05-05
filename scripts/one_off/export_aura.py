import os
import shutil
import argparse

def is_architecture_file(path):
    # Only allow python source files and config files
    if not (path.endswith('.py') or path.endswith('.yaml') or path.endswith('.toml')):
        return False
        
    parts = path.split(os.sep)
    
    # Exclude massive data, test, and environment directories
    excluded_dirs = {
        '.venv', '.git', '__pycache__', '.claude', 'tests', 
        '.mypy_cache', '.aura_architect', 'models', 'training', 
        'data', 'logs', 'assets', 'frontend', 'UI'
    }
    
    if any(part in excluded_dirs for part in parts):
        return False
        
    return True

def export_source(root_dir, output_dir, char_limit=4000000, copy_limit=1000):
    txt_output_dir = os.path.join(output_dir, 'Aura_Source_TXT')
    files_output_dir = os.path.join(output_dir, 'Aura_Source_Files')
    os.makedirs(txt_output_dir, exist_ok=True)
    os.makedirs(files_output_dir, exist_ok=True)
    
    current_txt_part = 1
    current_char_count = 0
    current_txt_file = None
    copied_files_count = 0
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if not is_architecture_file(filepath):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue
                
            # Handle text export
            header = f"\n{'='*80}\nFILE: {filepath.replace(root_dir, '')}\n{'='*80}\n"
            content_to_write = header + content
            
            if current_txt_file is None or current_char_count + len(content_to_write) > char_limit:
                if current_txt_file:
                    current_txt_file.close()
                current_txt_file = open(os.path.join(txt_output_dir, f'aura_source_part_{current_txt_part}.txt'), 'w', encoding='utf-8')
                current_txt_part += 1
                current_char_count = 0
                
            current_txt_file.write(content_to_write)
            current_char_count += len(content_to_write)
            
            # Handle file copy
            if copied_files_count < copy_limit:
                rel_path = os.path.relpath(filepath, root_dir)
                dest_path = os.path.join(files_output_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(filepath, dest_path)
                copied_files_count += 1
                
    if current_txt_file:
        current_txt_file.close()
        
    print(f"Exported text to {txt_output_dir} in {current_txt_part - 1} parts.")
    print(f"Copied {copied_files_count} files to {files_output_dir}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    export_source(args.root, args.output_dir)
