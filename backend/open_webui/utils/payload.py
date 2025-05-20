from open_webui.utils.task import prompt_template, prompt_variables_template
from open_webui.utils.misc import add_or_update_system_message
from typing import Callable, Optional
import sqlite3
import json
import pprint

DB_PATH = "/app/backend/data/webui.db"

def get_banned_words(user_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT data
            FROM feedback
            WHERE user_id = ?
              AND type = 'rating'
              AND json_extract(data, '$.rating') = -1
              AND json_extract(data, '$.comment') LIKE '%dont use the word%'
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        banned = []
        for row in rows:
            comment = json.loads(row[0]).get("comment", "").lower()
            if "dont use the word" in comment:
                word = comment.split("dont use the word")[-1].strip(" .\n\"'")
                banned.append(word)

        return banned

    except Exception as e:
        print("Error fetching banned words from DB:", e)
        return []


def get_tone_instructions(user_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT data
            FROM feedback
            WHERE user_id = ?
              AND type = 'rating'
              AND json_extract(data, '$.rating') = -1
              AND json_extract(data, '$.comment') LIKE '%tone%'
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        tones = []
        for row in rows:
            comment = json.loads(row[0]).get("comment", "")
            tones.append(comment)

        return tones

    except Exception as e:
        print("Error fetching tone instructions from DB:", e)
        return []


def apply_model_system_prompt_to_body(params: dict, form_data: dict, metadata: Optional[dict] = None, user=None) -> dict:
    system = params.get("system", None)
    if not system:
        return form_data

    if metadata:
        variables = metadata.get("variables", {})
        if variables:
            system = prompt_variables_template(system, variables)

    if user and hasattr(user, "id"):
        banned_words = get_banned_words(user.id)
        if banned_words:
            banned_line = f"Please do not use the following words in your response: {', '.join(banned_words)}."
            system = f"{system}\n\n{banned_line}"

        tone_instructions = get_tone_instructions(user.id)
        if tone_instructions:
            tone_note = " ".join(tone_instructions)
            system = f"{system}\n\n{tone_note}"

    if user:
        template_params = {
            "user_name": user.name,
            "user_location": user.info.get("location") if user.info else None,
        }
    else:
        template_params = {}

    system = prompt_template(system, **template_params)
    
    pp = pprint.PrettyPrinter(indent=2)

    print("\n==== DEBUG: Final Prompt Injection ====")
    pp.pprint({
    "Final System Prompt": system,
    "User ID": user.id if user and hasattr(user, "id") else None,
    "Params": params,
    "Tone Instructions": get_tone_instructions(user.id) if user and hasattr(user, "id") else [],
    "Banned Words": get_banned_words(user.id) if user and hasattr(user, "id") else [],
})

    form_data["messages"] = add_or_update_system_message(system, form_data.get("messages", []))
    return form_data


def apply_model_params_to_body(params: dict, form_data: dict, mappings: dict[str, Callable]) -> dict:
    if not params:
        return form_data

    for key, cast_func in mappings.items():
        if (value := params.get(key)) is not None:
            form_data[key] = cast_func(value)

    return form_data


def apply_model_params_to_body_openai(params: dict, form_data: dict) -> dict:
    mappings = {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
        "frequency_penalty": float,
        "presence_penalty": float,
        "reasoning_effort": str,
        "seed": lambda x: x,
        "stop": lambda x: [bytes(s, "utf-8").decode("unicode_escape") for s in x],
        "logit_bias": lambda x: x,
        "response_format": dict,
    }
    return apply_model_params_to_body(params, form_data, mappings)


def apply_model_params_to_body_ollama(params: dict, form_data: dict) -> dict:
    name_differences = {
        "max_tokens": "num_predict",
    }

    for key, value in name_differences.items():
        if (param := params.get(key, None)) is not None:
            params[value] = params[key]
            del params[key]

    mappings = {
        "temperature": float,
        "top_p": float,
        "seed": lambda x: x,
        "mirostat": int,
        "mirostat_eta": float,
        "mirostat_tau": float,
        "num_ctx": int,
        "num_batch": int,
        "num_keep": int,
        "num_predict": int,
        "repeat_last_n": int,
        "top_k": int,
        "min_p": float,
        "typical_p": float,
        "repeat_penalty": float,
        "presence_penalty": float,
        "frequency_penalty": float,
        "penalize_newline": bool,
        "stop": lambda x: [bytes(s, "utf-8").decode("unicode_escape") for s in x],
        "numa": bool,
        "num_gpu": int,
        "main_gpu": int,
        "low_vram": bool,
        "vocab_only": bool,
        "use_mmap": bool,
        "use_mlock": bool,
        "num_thread": int,
    }

    if "options" in form_data and "keep_alive" in form_data["options"]:
        form_data["keep_alive"] = form_data["options"]["keep_alive"]
        del form_data["options"]["keep_alive"]

    if "options" in form_data and "format" in form_data["options"]:
        form_data["format"] = form_data["options"]["format"]
        del form_data["options"]["format"]

    return apply_model_params_to_body(params, form_data, mappings)


def convert_messages_openai_to_ollama(messages: list[dict]) -> list[dict]:
    ollama_messages = []

    for message in messages:
        new_message = {"role": message["role"]}
        content = message.get("content", [])
        tool_calls = message.get("tool_calls", None)
        tool_call_id = message.get("tool_call_id", None)

        if isinstance(content, str) and not tool_calls:
            new_message["content"] = content
            if tool_call_id:
                new_message["tool_call_id"] = tool_call_id

        elif tool_calls:
            ollama_tool_calls = []
            for tool_call in tool_calls:
                ollama_tool_call = {
                    "index": tool_call.get("index", 0),
                    "id": tool_call.get("id", None),
                    "function": {
                        "name": tool_call.get("function", {}).get("name", ""),
                        "arguments": json.loads(tool_call.get("function", {}).get("arguments", "{}")),
                    },
                }
                ollama_tool_calls.append(ollama_tool_call)
            new_message["tool_calls"] = ollama_tool_calls
            new_message["content"] = ""

        else:
            content_text = ""
            images = []
            for item in content:
                if item.get("type") == "text":
                    content_text += item.get("text", "")
                elif item.get("type") == "image_url":
                    img_url = item.get("image_url", {}).get("url", "")
                    if img_url.startswith("data:"):
                        img_url = img_url.split(",")[-1]
                    images.append(img_url)
            if content_text:
                new_message["content"] = content_text.strip()
            if images:
                new_message["images"] = images

        ollama_messages.append(new_message)

    return ollama_messages


def convert_payload_openai_to_ollama(openai_payload: dict) -> dict:
    ollama_payload = {
        "model": openai_payload.get("model"),
        "messages": convert_messages_openai_to_ollama(openai_payload.get("messages")),
        "stream": openai_payload.get("stream", False),
    }

    if "tools" in openai_payload:
        ollama_payload["tools"] = openai_payload["tools"]

    if "format" in openai_payload:
        ollama_payload["format"] = openai_payload["format"]

    if openai_payload.get("options"):
        ollama_payload["options"] = openai_payload["options"]
        ollama_options = ollama_payload["options"]

        if "max_tokens" in ollama_options:
            ollama_options["num_predict"] = ollama_options.pop("max_tokens")

        if "system" in ollama_options:
            ollama_payload["system"] = ollama_options.pop("system")

        if "keep_alive" in ollama_options:
            ollama_payload["keep_alive"] = ollama_options.pop("keep_alive")

    if "stop" in openai_payload:
        ollama_payload.setdefault("options", {})["stop"] = openai_payload["stop"]

    if "metadata" in openai_payload:
        ollama_payload["metadata"] = openai_payload["metadata"]

    if "response_format" in openai_payload:
        response_format = openai_payload["response_format"]
        format_type = response_format.get("type")
        schema = response_format.get(format_type)
        if schema:
            ollama_payload["format"] = schema.get("schema")

    return ollama_payload
