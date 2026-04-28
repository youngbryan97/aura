import csv
import re
from pathlib import Path

def main():
    ledger_file = Path("/Users/bryan/Downloads/aura_exhaustive_forensic_ledger/by_category/assert_in_production.csv")
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
            
            if filepath not in file_mods:
                try:
                    file_mods[filepath] = {"lines": filepath.read_text().splitlines(), "changed": False}
                except Exception:
                    continue

            lines = file_mods[filepath]["lines"]
            if line_no < 0 or line_no >= len(lines):
                continue
                
            original_line = lines[line_no]
            
            # Simple assert with message: assert condition, "message"
            match = re.search(r'^([ \t]*)assert (.*?), (.*)$', original_line)
            if match:
                indent = match.group(1)
                condition = match.group(2)
                msg = match.group(3)
                lines[line_no] = f"{indent}if not ({condition}): raise RuntimeError({msg})"
                file_mods[filepath]["changed"] = True
                continue
                
            # Simple assert without message: assert condition
            match = re.search(r'^([ \t]*)assert (.*)$', original_line)
            if match:
                indent = match.group(1)
                condition = match.group(2)
                lines[line_no] = f"{indent}if not ({condition}): raise RuntimeError('Assertion failed')"
                file_mods[filepath]["changed"] = True

    for filepath, data in file_mods.items():
        if data["changed"]:
            content = "\n".join(data["lines"]) + "\n"
            filepath.write_text(content)
            print(f"Migrated P3 asserts in {filepath}")

if __name__ == "__main__":
    main()
