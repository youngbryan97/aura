import csv
import re
from pathlib import Path

def main():
    ledger_file = Path("/Users/bryan/Downloads/aura_exhaustive_forensic_ledger/by_category/direct_state_mutation_surface.csv")
    if not ledger_file.exists():
        print("CSV not found.")
        return

    # Track modifications per file
    file_mods = {}

    with open(ledger_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = Path(row["file"])
            if not filepath.exists():
                continue
                
            line_no = int(row["line"]) - 1  # 0-indexed
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
            
            # Simple assignments
            match1 = re.search(r'^([ \t]*)self\.state\.([a-zA-Z0-9_]+)[ \t]*=[ \t]*(.*)$', original_line)
            # Dict assignments
            match2 = re.search(r'^([ \t]*)self\.state\[[\'"]([a-zA-Z0-9_]+)[\'"]\][ \t]*=[ \t]*(.*)$', original_line)
            
            match = match1 or match2
            if match:
                indent = match.group(1)
                key = match.group(2)
                val = match.group(3)
                
                # We wrap the gateway call in create_task so it works synchronously or asynchronously
                new_line = f"{indent}get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='{key}', new_value={val}, cause='{symbol}')))"
                lines[line_no] = new_line
                file_mods[filepath]["changed"] = True

    for filepath, data in file_mods.items():
        if data["changed"]:
            content = "\n".join(data["lines"]) + "\n"
            
            # Inject required imports if not present
            if "StateMutationRequest" not in content and "get_state_gateway" not in content:
                content = "from core.state.state_gateway import get_state_gateway\nfrom core.runtime.gateways import StateMutationRequest\nfrom core.utils.task_tracker import get_task_tracker\n" + content
            
            filepath.write_text(content)
            print(f"Rewrote state mutations in {filepath}")

if __name__ == "__main__":
    main()
