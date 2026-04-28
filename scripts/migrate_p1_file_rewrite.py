import csv
import re
from pathlib import Path

def main():
    ledger_file = Path("/Users/bryan/Downloads/aura_exhaustive_forensic_ledger/by_category/direct_file_mutation.csv")
    if not ledger_file.exists():
        print("CSV not found.")
        return

    file_mods = {}

    with open(ledger_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = Path(row["file"])
            if not filepath.exists():
                continue
                
            line_no = int(row["line"]) - 1
            symbol = row["symbol"]
            
            if filepath not in file_mods:
                try:
                    file_mods[filepath] = {"lines": filepath.read_text().splitlines(), "changed": False}
                except Exception:
                    continue

            lines = file_mods[filepath]["lines"]
            if line_no < 0 or line_no >= len(lines):
                continue
                
            original_line = lines[line_no]
            
            # Simple unlinks: p.unlink() or lock_file.unlink() -> get_task_tracker().create_task(get_storage_gateway().delete(p))
            match = re.search(r'^([ \t]*)try:[ \t]*([a-zA-Z0-9_\.]+)\.unlink\([^\)]*\)[ \t]*$', original_line)
            if match:
                indent = match.group(1)
                obj = match.group(2)
                lines[line_no] = f"{indent}get_task_tracker().create_task(get_storage_gateway().delete({obj}, cause='{symbol}'))"
                file_mods[filepath]["changed"] = True
                continue
                
            match = re.search(r'^([ \t]*)([a-zA-Z0-9_\.]+)\.unlink\([^\)]*\)[ \t]*$', original_line)
            if match:
                indent = match.group(1)
                obj = match.group(2)
                lines[line_no] = f"{indent}get_task_tracker().create_task(get_storage_gateway().delete({obj}, cause='{symbol}'))"
                file_mods[filepath]["changed"] = True
                continue

            # Rmtree: shutil.rmtree(p) -> get_storage_gateway().delete_tree(p)
            match = re.search(r'^([ \t]*)(?:try:[ \t]*)?shutil\.rmtree\(([^)]+)\)[ \t]*$', original_line)
            if match:
                indent = match.group(1)
                obj = match.group(2)
                lines[line_no] = f"{indent}get_task_tracker().create_task(get_storage_gateway().delete_tree({obj}, cause='{symbol}'))"
                file_mods[filepath]["changed"] = True
                continue

            # Mkdir: p.mkdir(...) -> get_storage_gateway().create_dir(p)
            match = re.search(r'^([ \t]*)([a-zA-Z0-9_\.]+)\.mkdir\([^\)]*\)[ \t]*$', original_line)
            if match:
                indent = match.group(1)
                obj = match.group(2)
                lines[line_no] = f"{indent}get_task_tracker().create_task(get_storage_gateway().create_dir({obj}, cause='{symbol}'))"
                file_mods[filepath]["changed"] = True
                continue
                
            # Rename: patch_file.rename(...)
            match = re.search(r'^([ \t]*)([a-zA-Z0-9_\.]+)\.rename\((.*)\)[ \t]*$', original_line)
            if match:
                indent = match.group(1)
                obj = match.group(2)
                target = match.group(3)
                lines[line_no] = f"{indent}get_task_tracker().create_task(get_storage_gateway().rename({obj}, {target}, cause='{symbol}'))"
                file_mods[filepath]["changed"] = True
                continue

    for filepath, data in file_mods.items():
        if data["changed"]:
            content = "\n".join(data["lines"]) + "\n"
            
            # Inject required imports if not present
            if "get_storage_gateway" not in content:
                content = "from core.state.storage_gateway import get_storage_gateway\nfrom core.utils.task_tracker import get_task_tracker\n" + content
            
            filepath.write_text(content)
            print(f"Rewrote file mutations in {filepath}")

if __name__ == "__main__":
    main()
