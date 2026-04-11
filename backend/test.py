import requests
import json
import re

url = "http://16.171.3.197:8000/chat"


def keep_before_first_asterisk(text: str) -> tuple[str, bool]:
    source = str(text or "")
    if not source:
        return "", False

    star_index = source.find("*")
    if star_index == -1:
        return source, False

    return source[:star_index], True

def hard_clean(text: str) -> str:
    # Keep only the first section generated before the first asterisk.
    text, _ = keep_before_first_asterisk(text)

    # remove ALL model junk patterns
    text = text.replace("*Reflects feelings*", "")
    text = text.replace("*Reflectsfeelings*", "")
    text = text.replace("*Asks follow-up question*", "")
    text = text.replace("*Asksfollow-upquestion*", "")

    # fix missing spaces between lower and uppercase (e.g., "you.It" -> "you. It")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # normalize spacing
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def ask_serenity(user_text):
    full_response = ""

    with requests.post(
        url,
        json={"text": user_text},
        stream=True
    ) as r:

        for line in r.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8").strip()

            if not decoded.startswith("data:"):
                continue

            data = decoded.replace("data:", "").strip()

            try:
                obj = json.loads(data)
            except:
                continue

            if obj.get("done"):
                break

            token = obj.get("token", "")
            if not token:
                continue

            visible_token, reached_first_asterisk = keep_before_first_asterisk(token)
            if visible_token:
                full_response += visible_token

                # Print only visible text before the first asterisk.
                print(visible_token, end="", flush=True)

            if reached_first_asterisk:
                break

    print("\n")
    
    # Run the heavy cleaning on the fully assembled string at the end
    return hard_clean(full_response)


final_output = ask_serenity("I lost my brother")