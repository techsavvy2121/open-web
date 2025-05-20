import sqlite3
import json
import re
from datetime import datetime

DB_PATH = "/app/backend/data/webui.db"
JSON_PATH = "/app/backend/data/banned_words.json"

def extract_banned_word(comment):
    match = re.search(r"(?:dont|don't|avoid)\s+(?:use\s+(?:the\s+)?(?:word\s+)?)?['\"]?(\w+)['\"]?", comment.lower())
    return match.group(1).strip() if match else None

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, data
        FROM feedback
        WHERE json_extract(data, '$.comment') IS NOT NULL
          AND json_extract(data, '$.comment') != ''
    """)

    banned_words_map = {}

    for user_id, data_str in cursor.fetchall():
        data = json.loads(data_str)
        comment = data.get("comment", "")
        word = extract_banned_word(comment)
        if word:
            banned_words_map.setdefault(user_id, {"banned_words": [], "last_updated": ""})
            if word not in banned_words_map[user_id]["banned_words"]:
                banned_words_map[user_id]["banned_words"].append(word)
                banned_words_map[user_id]["last_updated"] = datetime.utcnow().isoformat()

    with open(JSON_PATH, "w") as f:
        json.dump(banned_words_map, f, indent=2)

    print("✅ Banned words extracted and saved to:", JSON_PATH)

if __name__ == "__main__":
    main()
