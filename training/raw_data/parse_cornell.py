import ast
import json

lines = {}
with open('/Users/bryan/.aura/live-source/training/raw_data/movie_lines.txt', 'r', encoding='iso-8859-1') as f:
    for line in f:
        parts = line.split(' +++$+++ ')
        if len(parts) == 5:
            lines[parts[0]] = parts[4].strip()

convs = []
with open('/Users/bryan/.aura/live-source/training/raw_data/movie_conversations.txt', 'r', encoding='iso-8859-1') as f:
    for line in f:
        parts = line.split(' +++$+++ ')
        if len(parts) == 4:
            convs.append(ast.literal_eval(parts[3].strip()))

parsed_conversations = []
for conv in convs:
    # We want pairs of (user, assistant) for training, so we look for conversations with 2 or more lines
    if len(conv) >= 2:
        # We can extract multiple pairs from a long conversation
        for i in range(len(conv) - 1):
            user_msg = lines.get(conv[i], '').strip()
            aura_msg = lines.get(conv[i+1], '').strip()
            if user_msg and aura_msg:
                parsed_conversations.append({
                    "user": user_msg,
                    "assistant": aura_msg
                })

# Save to json
with open('/Users/bryan/.aura/live-source/training/raw_data/human_conversations.json', 'w') as f:
    json.dump(parsed_conversations, f, indent=2)

print(f"Extracted {len(parsed_conversations)} conversation pairs from Cornell Movie Dialogs.")
