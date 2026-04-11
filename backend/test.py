import requests
import json
import re

url = "http://16.171.3.197:8000/chat"


def strip_starred_segments(text: str, preserve_edges: bool = False) -> str:
    source = str(text or "")
    if not source:
        return ""

    output_chars = []
    in_starred_segment = False
    index = 0
    length = len(source)

    while index < length:
        char = source[index]
        if char == "*":
            while index < length and source[index] == "*":
                index += 1
            in_starred_segment = not in_starred_segment
            continue

        if not in_starred_segment:
            output_chars.append(char)
        index += 1

    cleaned = "".join(output_chars)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned if preserve_edges else cleaned.strip()


def clean_stream_token(token: str, in_starred_segment: bool) -> tuple[str, bool]:
    source = str(token or "")
    if not source:
        return "", in_starred_segment

    output_chars = []
    index = 0
    length = len(source)

    while index < length:
        char = source[index]
        if char == "*":
            while index < length and source[index] == "*":
                index += 1
            in_starred_segment = not in_starred_segment
            continue

        if not in_starred_segment:
            output_chars.append(char)
        index += 1

    cleaned = "".join(output_chars)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned, in_starred_segment


def hard_clean(text: str) -> str:
    text = strip_starred_segments(text, preserve_edges=True)

    # Fix missing spaces between lower and uppercase (e.g., "you.It" -> "you. It")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Normalize spacing (removes double spaces left behind by the regex)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def ask_serenity(user_text):
    full_response = ""
    in_starred_segment = False

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

            clean_token, in_starred_segment = clean_stream_token(token, in_starred_segment)
            if not clean_token:
                continue

            full_response += clean_token

            # Print only cleaned token text, while preserving incoming spacing.
            print(clean_token, end="", flush=True)

    print("\n")
    
    # Run the heavy cleaning on the fully assembled string at the end
    return hard_clean(full_response)


final_output = ask_serenity("I lost my brother")