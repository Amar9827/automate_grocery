import os
import logging
import time

import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
    http_client=httpx.Client(verify=False),
)
MODEL = "llama-3.3-70b-versatile"


def _call_groq(messages, max_tokens=300):
    """Call Groq with retry on 429/503."""
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL, max_tokens=max_tokens, messages=messages
            )
        except Exception as e:
            err = str(e)
            if attempt < 2 and ("429" in err or "503" in err or "rate" in err.lower()):
                logger.warning(f"Groq retry {attempt+1}: {err[:80]}")
                time.sleep(3)
            else:
                raise

def parse_grocery_intent(text: str) -> dict:
    """
    Extract grocery items from a free-text message.

    Returns:
        {
            "is_grocery": bool,
            "items": ["milk", "eggs", "bread"],   # item names only (no quantities)
            "raw_text": str
        }
    """
    response = _call_groq([
            {
                "role": "system",
                "content": (
                    "Extract grocery items from the user's message. "
                    "Respond with ONLY the item names, one per line. "
                    "KEEP brand names if mentioned (e.g. 'Dove shampoo', 'Amul butter', 'Nandini milk'). "
                    "Do NOT include quantities, units, or pack sizes "
                    "(e.g. output 'milk' not 'milk 1L', output 'eggs' not 'eggs 6pk'). "
                    "If the message is NOT a grocery request, respond with exactly: NOT_GROCERY"
                )
            },
            {"role": "user", "content": text}
        ])

    raw = response.choices[0].message.content.strip()
    logger.info(f"Intent parse response: {raw[:100]}")

    if raw == "NOT_GROCERY":
        return {"is_grocery": False, "items": [], "raw_text": text}

    items = [line.strip() for line in raw.split("\n") if line.strip()]
    return {"is_grocery": True, "items": items, "raw_text": text}


def reconcile_with_preferences(items: list[str], members: list[dict]) -> dict:
    """
    Check items against household dietary preferences.
    Returns { "safe_items": [...], "flagged": [{"item": ..., "reason": ...}] }
    """
    if not members:
        return {"safe_items": items, "flagged": []}

    # Build preference summary
    prefs_text = "\n".join(
        f"- {m['name']}: {m['preferences']}"
        for m in members
        if m.get("preferences")
    )

    if not prefs_text:
        return {"safe_items": items, "flagged": []}

    items_text = "\n".join(items)

    response = _call_groq([
            {
                "role": "system",
                "content": (
                    "Check grocery items against household dietary restrictions. "
                    "For each conflicting item, output: FLAG: <item> | <member name> | <reason> | <substitute>\n"
                    "For safe items, output: OK: <item>\n"
                    "One line per item. No other text."
                )
            },
            {
                "role": "user",
                "content": f"Household preferences:\n{prefs_text}\n\nItems to check:\n{items_text}"
            }
        ], max_tokens=400)

    raw    = response.choices[0].message.content.strip()
    safe   = []
    flagged = []

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("OK: "):
            safe.append(line[4:].strip())
        elif line.startswith("FLAG: "):
            parts = line[6:].split("|")
            if len(parts) >= 3:
                flagged.append({
                    "item":       parts[0].strip(),
                    "member":     parts[1].strip(),
                    "reason":     parts[2].strip(),
                    "substitute": parts[3].strip() if len(parts) > 3 else "",
                })

    return {"safe_items": safe, "flagged": flagged}
