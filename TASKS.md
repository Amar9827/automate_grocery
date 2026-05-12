# Grocery Bot — Build Tasks

## Context

A Telegram bot that lets household members order groceries from Zepto via text or voice.
The bot transcribes voice notes, parses grocery intent using an LLM, reconciles household
dietary preferences, builds a Zepto cart via MCP, and confirms before placing the order.

**Current state:** Task 1.1 complete — basic bot echo works.
**Stack:** Python, python-telegram-bot v20+, faster-whisper, Groq API, Zepto MCP (existing zepto_client.py)

---

## File structure (target)

```
grocery_bot/
├── bot.py              # Telegram bot — entry point, all handlers
├── db.py               # SQLite — members, preferences, order history
├── transcriber.py      # faster-whisper voice transcription
├── llm.py              # Groq API — intent parsing, preference reconciliation
├── zepto.py            # Zepto MCP session — lifted from zepto_client.py
├── requirements.txt
└── .env
```

---

## Phase 1 — Telegram bot foundation

### Task 1.1 ✅ DONE
Basic echo bot with text and voice handlers. `/start` and `/help` commands.

---

### Task 1.2 — Household member registration

**File:** `db.py`, `bot.py`

Create a SQLite database with a `members` table and wire up a `/register` command.

**db.py — create:**
```python
import sqlite3, os

DB_PATH = os.environ.get("DB_PATH", "grocery_bot.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS members (
                telegram_id   INTEGER PRIMARY KEY,
                name          TEXT    NOT NULL,
                household_id  TEXT    NOT NULL DEFAULT 'default',
                preferences   TEXT    DEFAULT '',
                joined_at     TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

def register_member(telegram_id: int, name: str, household_id: str = "default"):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO members (telegram_id, name, household_id)
            VALUES (?, ?, ?)
            ON CONFLICT(telegram_id) DO UPDATE SET name=excluded.name
        """, (telegram_id, name, household_id))
        conn.commit()

def get_member(telegram_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM members WHERE telegram_id = ?", (telegram_id,)
        ).fetchone()
        return dict(row) if row else None

def get_household_members(household_id: str = "default") -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM members WHERE household_id = ?", (household_id,)
        ).fetchall()
        return [dict(r) for r in rows]
```

**bot.py — add `/register` handler:**
```python
from db import init_db, register_member, get_member

# Call init_db() before app.run_polling()

async def cmd_register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    args = context.args  # words after /register

    if not args:
        await update.message.reply_text(
            "Usage: /register <your name>\nExample: /register Amar"
        )
        return

    name = " ".join(args).strip()
    register_member(user.id, name)

    await update.message.reply_text(
        f"✅ Registered as *{name}*!\n\n"
        f"Next, set your dietary preferences:\n"
        f"/preferences vegetarian, no gluten",
        parse_mode="Markdown"
    )

# Register handler:
app.add_handler(CommandHandler("register", cmd_register))
```

**Test:** Send `/register Amar` — bot should confirm registration.
Run again with same Telegram ID — name should update, not duplicate.

---

### Task 1.3 — Dietary preferences per member

**File:** `db.py`, `bot.py`

Add `/preferences` and `/myprefs` commands.

**db.py — add:**
```python
def set_preferences(telegram_id: int, preferences: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE members SET preferences = ? WHERE telegram_id = ?",
            (preferences, telegram_id)
        )
        conn.commit()
```

**bot.py — add handlers:**
```python
async def cmd_preferences(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    member = get_member(user.id)

    if not member:
        await update.message.reply_text("Please /register first.")
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: /preferences <restrictions>\n"
            "Example: /preferences vegetarian, no onion, lactose intolerant"
        )
        return

    prefs = " ".join(context.args).strip()
    set_preferences(user.id, prefs)

    await update.message.reply_text(
        f"✅ Preferences saved: _{prefs}_\n\n"
        f"These will be checked before adding items to cart.",
        parse_mode="Markdown"
    )


async def cmd_myprefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    member = get_member(user.id)

    if not member:
        await update.message.reply_text("Please /register first.")
        return

    prefs = member.get("preferences", "")
    if prefs:
        await update.message.reply_text(f"Your preferences: _{prefs}_", parse_mode="Markdown")
    else:
        await update.message.reply_text("No preferences set. Use /preferences to add some.")


async def cmd_clearprefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    set_preferences(user.id, "")
    await update.message.reply_text("✅ Preferences cleared.")

# Register handlers:
app.add_handler(CommandHandler("preferences", cmd_preferences))
app.add_handler(CommandHandler("myprefs",     cmd_myprefs))
app.add_handler(CommandHandler("clearprefs",  cmd_clearprefs))
```

**Test:** `/preferences vegetarian, no nuts` → confirm saved.
`/myprefs` → should show them back.
`/clearprefs` → should clear them.

---

## Phase 2 — Voice transcription

### Task 2.1 — Download Telegram voice notes

**File:** `bot.py`

Update `on_voice_message` to download the `.ogg` file to a temp path.

```python
import tempfile, os

async def on_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user     = update.effective_user
    username = user.first_name or f"User {user.id}"
    voice    = update.message.voice

    await update.message.reply_text("🎙 Transcribing...")

    # Download voice note to temp file
    voice_file = await context.bot.get_file(voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    await voice_file.download_to_drive(tmp_path)
    logger.info(f"Voice note saved to {tmp_path} ({voice.duration}s)")

    # Phase 2.2 will transcribe here — for now confirm download
    await update.message.reply_text(
        f"✅ Downloaded ({voice.duration}s). Transcription coming next!"
    )

    # Clean up temp file
    os.unlink(tmp_path)
```

**Test:** Send a voice note — bot should reply "Transcribing..." then confirm download.

---

### Task 2.2 — Transcribe with faster-whisper

**File:** `transcriber.py` (new), `bot.py`

**transcriber.py — create:**
```python
import logging, os
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Load once at module level — expensive to reload per request
# "small" model: good balance of speed and accuracy for Indian English
# Change to "medium" for better accuracy at the cost of speed
_model = None

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        model_size = os.environ.get("WHISPER_MODEL", "small")
        logger.info(f"Loading Whisper model: {model_size}")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("Whisper model loaded")
    return _model

def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file to text.
    Returns empty string if transcription fails or produces no output.
    """
    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        language="en",           # Set to None for auto-detect if you use other languages
        beam_size=5,
        vad_filter=True,         # Skip silent segments — reduces hallucination
        vad_parameters={"min_silence_duration_ms": 500},
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    logger.info(f"Transcribed ({info.duration:.1f}s): {text[:100]}")
    return text
```

**bot.py — update `on_voice_message`:**
```python
from transcriber import transcribe

async def on_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user     = update.effective_user
    username = user.first_name or f"User {user.id}"

    await update.message.reply_text("🎙 Transcribing...")

    voice_file = await context.bot.get_file(update.message.voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await voice_file.download_to_drive(tmp_path)
        text = transcribe(tmp_path)

        if not text:
            await update.message.reply_text(
                "Sorry, I couldn't make out what you said. Please try again or type your order."
            )
            return

        # Route to the same handler as text messages
        await handle_message(user.id, username, text, update, context)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Add to requirements.txt:**
```
faster-whisper==1.0.3
```

**Test:** Send a voice note saying "add eggs and bread".
Bot should transcribe it and echo back the text.

---

### Task 2.3 — Unify text and voice ✅ (done by 2.2)

Both paths already call `handle_message()` with a plain text string.
No additional work needed — the pipeline is unified.

---

## Phase 3 — LLM intent parsing and preference reconciliation

### Task 3.1 — Parse grocery intent from free text

**File:** `llm.py` (new)

```python
import os, logging
from groq import Groq

logger = logging.getLogger(__name__)
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "llama-3.3-70b-versatile"

def parse_grocery_intent(text: str) -> dict:
    """
    Extract grocery items from a free-text message.

    Returns:
        {
            "is_grocery": bool,
            "items": ["milk 1L", "eggs 6pk", "bread"],
            "raw_text": str
        }
    """
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract grocery items from the user's message. "
                    "Respond with ONLY the items, one per line. "
                    "Include quantities if mentioned (e.g. 'milk 1L', 'eggs 6'). "
                    "If the message is NOT a grocery request, respond with exactly: NOT_GROCERY"
                )
            },
            {"role": "user", "content": text}
        ]
    )

    raw = response.choices[0].message.content.strip()
    logger.info(f"Intent parse response: {raw[:100]}")

    if raw == "NOT_GROCERY":
        return {"is_grocery": False, "items": [], "raw_text": text}

    items = [line.strip() for line in raw.split("\n") if line.strip()]
    return {"is_grocery": True, "items": items, "raw_text": text}
```

**bot.py — update `handle_message`:**
```python
from llm import parse_grocery_intent

async def handle_message(user_id, username, text, update, context):
    intent = parse_grocery_intent(text)

    if not intent["is_grocery"]:
        await update.message.reply_text(
            "That doesn't look like a grocery request. Try something like:\n"
            "_\"add milk, eggs and bread\"_ or _\"ingredients for pasta\"_",
            parse_mode="Markdown"
        )
        return

    items_list = "\n".join(f"• {item}" for item in intent["items"])
    await update.message.reply_text(
        f"Got it! I'll search for:\n{items_list}\n\n_(Cart building coming in Phase 4)_",
        parse_mode="Markdown"
    )
```

**Test:** Send "add eggs and Amul butter" → should list the items.
Send "what's the weather" → should reply with "doesn't look like a grocery request".

---

### Task 3.2 — Household preference reconciliation

**File:** `llm.py`

Add a function that checks parsed items against all household members' preferences.

```python
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

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        messages=[
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
        ]
    )

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
```

**bot.py — update `handle_message` to include reconciliation:**
```python
from db import get_member, get_household_members
from llm import parse_grocery_intent, reconcile_with_preferences

async def handle_message(user_id, username, text, update, context):
    member = get_member(user_id)
    if not member:
        await update.message.reply_text(
            "Please /register first before placing orders."
        )
        return

    intent = parse_grocery_intent(text)
    if not intent["is_grocery"]:
        await update.message.reply_text(
            "That doesn't look like a grocery request. Try:\n"
            "_\"add milk, eggs and bread\"_",
            parse_mode="Markdown"
        )
        return

    # Check against household preferences
    household = get_household_members(member["household_id"])
    result    = reconcile_with_preferences(intent["items"], household)

    reply = ""

    if result["flagged"]:
        flags = "\n".join(
            f"⚠️ *{f['item']}* — conflicts with {f['member']}'s preferences "
            f"({f['reason']})"
            + (f"\n   Substitute: _{f['substitute']}_" if f['substitute'] else "")
            for f in result["flagged"]
        )
        reply += f"*Preference conflicts:*\n{flags}\n\n"

    if result["safe_items"]:
        safe = "\n".join(f"✅ {item}" for item in result["safe_items"])
        reply += f"*Ready to add:*\n{safe}"

    reply += "\n\n_(Zepto cart building coming in Phase 4)_"
    await update.message.reply_text(reply, parse_mode="Markdown")
```

**Test:** Register two household members with different preferences.
Send "add chicken and milk" — chicken should be flagged for the vegetarian member.

---

### Task 3.3 — Switch to Groq in zepto_client.py

**File:** `zepto_client.py`

Update the OpenAI client to point at Groq:

```python
from groq import Groq

# Replace the existing client setup:
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "llama-3.3-70b-versatile"

# The rest of zepto_client.py stays the same —
# Groq uses the same OpenAI-compatible interface
```

**Test:** Run zepto_client.py standalone — confirm it still connects to Zepto MCP and can search products.

---

## Phase 4 — Zepto MCP cart builder

### Task 4.1 — Wire Zepto MCP into the bot

**File:** `zepto.py` (new), `bot.py`

Lift the MCP session setup from `zepto_client.py` into a reusable module.

**zepto.py — create:**
```python
import asyncio, json, os, logging
import httpx
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "llama-3.3-70b-versatile"

ZEPTO_MCP_URL = os.environ.get("ZEPTO_MCP_URL", "https://mcp.zepto.co.in/mcp")

def to_openai_tools(mcp_tools) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            }
        }
        for t in mcp_tools.tools
    ]

async def build_cart(items: list[str], system_context: str = "") -> str:
    """
    Connect to Zepto MCP and add items to cart.
    Returns a summary string of what was added.

    Args:
        items:          List of grocery items to search and add
        system_context: Additional context for the LLM (e.g. brand preferences)
    """
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(ZEPTO_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools    = await session.list_tools()
            oa_tools = to_openai_tools(tools)

            # Build the cart via multi-turn LLM + tool calling
            system_prompt = (
                "You are a grocery shopping assistant for Zepto. "
                "Search for each item, pick the best match, and add it to the cart. "
                "After adding all items, call get_cart to get the final cart summary. "
                "Be efficient — search and add one item at a time. "
                + system_context
            )

            items_text = "\n".join(f"- {item}" for item in items)
            history    = [{"role": "user", "content": f"Add these items to my cart:\n{items_text}"}]

            for _ in range(20):  # Max 20 tool calls
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system_prompt}] + history,
                    tools=oa_tools,
                    max_tokens=800,
                )
                msg = response.choices[0].message

                if not msg.tool_calls:
                    return msg.content or "Cart updated."

                history.append(msg)

                for tc in msg.tool_calls:
                    try:
                        args   = json.loads(tc.function.arguments)
                        result = await session.call_tool(tc.function.name, args)
                        content = " ".join(
                            c.text for c in result.content
                            if hasattr(c, "text")
                        )
                        history.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      content,
                        })
                    except Exception as e:
                        logger.error(f"Tool call error: {e}")
                        history.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      f"Error: {e}",
                        })

    return "Cart build complete."
```

**bot.py — update `handle_message` to call `build_cart`:**
```python
from zepto import build_cart

async def handle_message(user_id, username, text, update, context):
    # ... (intent parse + reconciliation from Phase 3) ...

    await update.message.reply_text("🔍 Searching Zepto...")

    try:
        summary = await build_cart(result["safe_items"])
        await update.message.reply_text(
            f"🛒 Items added to cart!\n\n{summary}\n\n"
            f"Reply *yes* to place the order or *no* to cancel.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Cart build failed: {e}")
        await update.message.reply_text(
            "Sorry, something went wrong with Zepto. Please try again."
        )
```

---

### Task 4.2 — Cart confirmation flow

**File:** `bot.py`

Use `ConversationHandler` to manage the confirm/cancel state.

```python
from telegram.ext import ConversationHandler

CONFIRM = 1  # Conversation state

async def handle_message(user_id, username, text, update, context):
    # ... (Phase 3 logic) ...

    summary = await build_cart(result["safe_items"])

    # Store summary in context for the confirmation step
    context.user_data["pending_summary"] = summary

    await update.message.reply_text(
        f"🛒 *Cart ready!*\n\n{summary}\n\n"
        f"Reply *yes* to place the order or *no* to cancel.",
        parse_mode="Markdown"
    )
    return CONFIRM


async def confirm_order(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply = update.message.text.lower().strip()

    if reply in ("yes", "y", "confirm", "ok", "place order"):
        await update.message.reply_text("⏳ Placing your order...")
        try:
            # Call checkout via Zepto MCP
            result = await place_order()
            await update.message.reply_text(f"✅ Order placed! {result}")
        except Exception as e:
            await update.message.reply_text(f"❌ Order failed: {e}")
    else:
        await update.message.reply_text("❌ Order cancelled. Cart cleared.")

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# Register ConversationHandler (replaces the text MessageHandler):
conv_handler = ConversationHandler(
    entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_message)],
    states={
        CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, confirm_order)]
    },
    fallbacks=[CommandHandler("cancel", cancel)],
)
app.add_handler(conv_handler)
```

---

### Task 4.3 — Order history integration

**File:** `db.py`, `zepto.py`

Before searching Zepto, check SQLite for the user's preferred variant of each item.

**db.py — add:**
```python
def get_preferred_variant(telegram_id: int, item_keyword: str) -> str | None:
    """
    Look up the most recently ordered product matching a keyword.
    Returns the product name string if found, None otherwise.
    """
    with get_conn() as conn:
        row = conn.execute("""
            SELECT product_name FROM order_history
            WHERE telegram_id = ?
            AND LOWER(product_name) LIKE ?
            ORDER BY ordered_at DESC
            LIMIT 1
        """, (telegram_id, f"%{item_keyword.lower()}%")).fetchone()
        return row["product_name"] if row else None
```

**zepto.py — update `build_cart`:**
```python
from db import get_preferred_variant

async def build_cart(items: list[str], telegram_id: int = None) -> str:
    # Before building prompt, enrich items with history
    enriched = []
    for item in items:
        if telegram_id:
            preferred = get_preferred_variant(telegram_id, item.split()[0])
            if preferred:
                enriched.append(f"{item} (previously ordered: {preferred})")
                continue
        enriched.append(item)

    # Pass enriched list to LLM as before
    ...
```

---

## Phase 5 — Polish and error handling

### Task 5.1 — Graceful error handling

Every failure path sends a clear Telegram message. No silent crashes.

Key scenarios to handle in `bot.py` and `zepto.py`:
- Zepto MCP connection timeout → "Zepto is slow right now, try again in a moment"
- Groq rate limit (429) → retry once after 3 seconds, then fail gracefully
- Voice note too short / empty transcription → "Couldn't make that out, please type your order"
- Item not found on Zepto → include in reply: "❓ Couldn't find: [item]"
- User not registered → prompt to /register before any cart action

---

### Task 5.2 — Bot commands

Add `/status` and `/history` commands.

**`/status`** — fetch current Zepto cart via MCP and format it:
```python
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cart = await get_current_cart()
    await update.message.reply_text(f"🛒 Your cart:\n{cart}")
```

**`/history`** — read last 3 orders from SQLite:
```python
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user    = update.effective_user
    orders  = get_recent_orders(user.id, limit=3)
    if not orders:
        await update.message.reply_text("No order history yet.")
        return
    text = "\n\n".join(
        f"📦 {o['ordered_at'][:10]}\n{o['summary']}"
        for o in orders
    )
    await update.message.reply_text(text)
```

---

## Environment variables (.env)

```
TELEGRAM_BOT_TOKEN=your_token_here
GROQ_API_KEY=gsk_...
ZEPTO_MCP_URL=https://mcp.zepto.co.in/mcp
WHISPER_MODEL=small
DB_PATH=grocery_bot.db
```

---

## Running the bot

```bash
pip install -r requirements.txt
python bot.py
```

---

## Key design decisions

- **All input → `handle_message()`** — text and voice both funnel to one function
- **Groq for all LLM work** — intent parsing, reconciliation, and Zepto cart building
- **SQLite for all persistence** — members, preferences, order history
- **ConversationHandler for confirm flow** — never places an order without user approval
- **Modular files** — `db.py`, `transcriber.py`, `llm.py`, `zepto.py` are all independently testable
