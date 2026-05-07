import requests
import json
import re
import urllib.parse
from bs4 import BeautifulSoup
import time

TARGETS = {
    "GLADOS": ["Portal (video game)", "Portal 2"],
    "BAYONETTA": ["Bayonetta"],
    "ANDROID_18": ["Dragon Ball Z", "Dragon Ball Super"],
    "JANE": ["Speaker for the Dead", "Xenocide", "Children of the Mind"]
}

CHARACTER_HINTS = {
    "GLADOS": ["GLaDOS"],
    "BAYONETTA": ["Bayonetta", "Cereza"],
    "ANDROID_18": ["18", "Android 18", "Lazuli"],
    "JANE": ["Jane"]
}


def fetch_wikiquote(title):
    url = f"https://en.wikiquote.org/w/api.php?action=parse&page={urllib.parse.quote(title)}&format=json"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            try:
                data = response.json()
                if 'parse' in data:
                    return data['parse']['text']['*']
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"Error fetching {title}: {e}")
    return None


def parse_html_for_character(html, character_name_hints):
    soup = BeautifulSoup(html, 'html.parser')
    quotes = []

    # Strategy 1
    for b in soup.find_all(['b', 'strong']):
        text = b.get_text().strip()
        is_match = any(hint.lower() in text.lower() for hint in character_name_hints)
        if is_match:
            parent = b.parent
            if parent.name in ['dd', 'li', 'p']:
                full_text = parent.get_text()
                quote_text = full_text.replace(text, '', 1).strip()
                if quote_text.startswith(':'):
                    quote_text = quote_text[1:].strip()
                quote_text = re.sub(r'\[.*?\]', '', quote_text).strip()
                if quote_text and len(quote_text) > 10:
                    quotes.append(quote_text)

    # Strategy 2
    for h_tag in soup.find_all(['h2', 'h3', 'h4']):
        text = h_tag.get_text().strip()
        is_match = any(hint.lower() in text.lower() for hint in character_name_hints)
        if is_match:
            next_node = h_tag.find_next_sibling()
            count = 0
            while next_node and next_node.name not in ['h2', 'h3', 'h4'] and count < 10:
                if next_node.name in ['ul', 'dl']:
                    for li in next_node.find_all(['li', 'dd']):
                        quote_text = li.get_text().strip()
                        quote_text = re.sub(r'\[.*?\]', '', quote_text).strip()
                        if quote_text and len(quote_text) > 10 and ':' not in quote_text[:15]:
                            quotes.append(quote_text)
                next_node = next_node.find_next_sibling()
                count += 1

    return list(set(quotes))

all_scraped_quotes = []

for char_id, pages in TARGETS.items():
    print(f"Scraping for {char_id}...")
    hints = CHARACTER_HINTS.get(char_id, [char_id])
    char_quotes = []

    for page in pages:
        html = fetch_wikiquote(page)
        if html:
            extracted = parse_html_for_character(html, hints)
            char_quotes.extend(extracted)
        time.sleep(0.5)

    char_quotes = list(set(char_quotes))
    print(f"  -> Found {len(char_quotes)} unique quotes for {char_id}.")

    # Generate some simple conversational contexts for the raw quotes
    for q in char_quotes:
        prompt = "..." if len(q) > 50 else "What did you say?"
        all_scraped_quotes.append({
            "user": prompt,
            "assistant": q,
            "source": char_id
        })

with open('/Users/bryan/.aura/live-source/training/raw_data/new_scraped_quotes.json', 'w') as f:
    json.dump(all_scraped_quotes, f, indent=2)

print("Saved to new_scraped_quotes.json")
