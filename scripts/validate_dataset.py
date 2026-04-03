import json
import re
import sys
from pathlib import Path

def validate_dataset(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {file_path} not found.")
        return False

    errors = []
    line_num = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                
                # 1. Check for required tags
                if "<thought>" not in text or "</thought>" not in text:
                    errors.append(f"Line {line_num}: Missing <thought> tags.")
                if "<action>" not in text or "</action>" not in text:
                    errors.append(f"Line {line_num}: Missing <action> tags.")
                
                # 2. Extract action content
                action_match = re.search(r"<action>\n?(.*?)\n?</action>", text, re.DOTALL)
                if not action_match:
                    errors.append(f"Line {line_num}: Could not extract <action> content.")
                    continue
                    
                action_content = action_match.group(1).strip()
                
                # 3. Check for legacy 'use_tool:' syntax
                if "use_tool:" in action_content:
                    errors.append(f"Line {line_num}: Legacy 'use_tool:' syntax detected.")
                
                # 4. JSON Tool Call Validation
                # Only validate as JSON if it contains the "tool" key in a structured way
                if '"tool":' in action_content:
                    # Find potential JSON start/end
                    try:
                        # Extract JSON object (assuming it's a standalone object or clearly demarcated)
                        # We look for the outermost braces if "tool" is inside
                        start_idx = action_content.find("{")
                        end_idx = action_content.rfind("}")
                        if start_idx != -1 and end_idx != -1:
                            json_str = action_content[start_idx:end_idx+1]
                            action_json = json.loads(json_str)
                            if "tool" not in action_json:
                                errors.append(f"Line {line_num}: Object with '{{' missing 'tool' key.")
                            if "params" not in action_json:
                                errors.append(f"Line {line_num}: JSON tool call missing 'params' key.")
                    except json.JSONDecodeError:
                        # If it contains "tool": it really should be valid JSON
                        errors.append(f"Line {line_num}: Snippet containing '\"tool\":' is not valid JSON.")
                        
            except json.JSONDecodeError:
                errors.append(f"Line {line_num}: Invalid JSON structure for line.")

    if errors:
        print(f"Validation failed for {file_path}:")
        for error in errors[:10]: # Cap output
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")
        return False
    
    print(f"✅ {file_path} passed validation.")
    return True

if __name__ == "__main__":
    target = str(Path.home() / ".aura" / "data" / "synthetic_training" / "lora_dataset.jsonl")
    if len(sys.argv) > 1:
        target = sys.argv[1]
    success = validate_dataset(target)
    sys.exit(0 if success else 1)
