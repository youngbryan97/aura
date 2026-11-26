import os
from pathlib import Path
from datetime import datetime

def export_source():
    from core.config import config
    project_root = config.paths.project_root
    desktop = Path.home() / "Desktop"
    output_file = desktop / "aura_source.txt"
    
    # Files/Dirs to exclude
    exclude_dirs = {
        ".git", ".venv", "venv", "__pycache__", "dist", "build", "node_modules", 
        "brain", ".next", ".DS_Store", "artifacts", "backups", "interface/static/dist",
        "data/browser_profile", "data/memories"
    }
    exclude_exts = {
        ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin", 
        ".jpg", ".png", ".ico", ".pdf", ".zip", ".tar.gz", ".mp3", ".wav",
        ".node", ".map", ".log", ".sqlite", ".db"
    }
    
    # Load .gitignore if present
    ignored_patterns = set()
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignored_patterns.add(line)
                    
    print(f"Exporting source from {project_root} to {output_file}...")
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("================================================================================\n")
        out.write("                       AURA: INFINITY — FULL SOURCE EXPORT                      \n")
        out.write(f"                       EXPORT DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("================================================================================\n\n")
        
        for root, dirs, files in os.walk(project_root):
            rel_root = Path(root).relative_to(project_root)
            rel_path_str = rel_root.as_posix()
            
            # Filter dirs
            dirs[:] = [d for d in dirs if d not in exclude_dirs and d not in ignored_patterns]
            
            for file in sorted(files):
                file_path = Path(root) / file
                if file_path.suffix.lower() in exclude_exts or file in exclude_dirs or file in ignored_patterns:
                    continue
                
                rel_path = file_path.relative_to(project_root)
                
                try:
                    # Skip files larger than 500KB to prevent bloat
                    if file_path.stat().st_size > 500 * 1024:
                        print(f"Skipping large file: {rel_path} ({file_path.stat().st_size / 1024:.1f} KB)")
                        continue

                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        
                    out.write(f"\n\n{'#' * 80}\n")
                    out.write(f"### FILE: {rel_path}\n")
                    out.write(f"{'#' * 80}\n\n")
                    out.write(content)
                    out.write("\n")
                    count += 1
                except (OSError, UnicodeError) as e:
                    print(f"Could not read {rel_path}: {e}")
    
    final_size = output_file.stat().st_size / (1024 * 1024)
    print(f"Export complete! {count} files bundled into {output_file}")
    print(f"Total File Size: {final_size:.2f} MB")

if __name__ == "__main__":
    export_source()
