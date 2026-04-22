import requests
import json
import re
import sys

# Force Windows console to accept UTF-8 characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

url = "http://51.21.162.77:8000/chat"

def hard_clean(text: str) -> str:
    # Fix glued words/sentences like "you.It" -> "you. It"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    
    # Normalize multiple spaces into a single space
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def ask_serenity(user_text):
    full_response = ""

    with requests.post(
        url,
        json={"text": user_text},
        stream=True
    ) as r:
        r.raise_for_status()

        for line in r.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8").strip()

            if not decoded.startswith("data:"):
                continue

            data = decoded.replace("data:", "", 1).strip()

            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            if obj.get("done"):
                break

            token = obj.get("token", "")
            if not token:
                continue

            # --- EFFICIENT CUTOFF LOGIC ---
            # Look for either an asterisk (action tags) or a hashtag (social artifacts)
            star_idx = token.find("*")
            hash_idx = token.find("#")
            
            # Find the earliest occurrence of either character
            indices = [i for i in (star_idx, hash_idx) if i != -1]
            
            if indices:
                cutoff_index = min(indices)
                clean_token = token[:cutoff_index]
                
                full_response += clean_token
                print(clean_token, end="", flush=True)
                
                # BREAK the loop completely to stop downloading junk!
                break
            else:
                # Normal token, just add and print
                full_response += token
                print(token, end="", flush=True)

    print("\n")
    return hard_clean(full_response)

if __name__ == "__main__":
    final_output = ask_serenity("I lost my brother")