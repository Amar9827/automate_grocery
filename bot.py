# bot.py — Task 1.1: Telegram Bot Foundation
#
# WHAT THIS FILE DOES:
#   Sets up a Telegram bot using python-telegram-bot v20+ (async Application pattern).
#   Handles text messages and voice notes, routing both through a single
#   handle_message() function that all future logic will plug into.
#
# HOW TO RUN:
#   pip install -r requirements.txt
#   Add TELEGRAM_BOT_TOKEN to your .env file
#   python bot.py
#
# HOW TO TEST:
#   Open Telegram → find your bot → send it a message
#   It should echo back: "You said: [your message]"

import logging
import os
import tempfile
from dotenv import load_dotenv

from db import (
    init_db, register_member, get_member, get_household_members,
    set_preferences, get_product_pref, set_product_pref,
    save_order, get_recent_orders,
)
from transcriber import transcribe
from llm import parse_grocery_intent, reconcile_with_preferences
from zepto import search_zepto, add_to_cart, place_order, clear_cart, preview_order, get_cart_status, check_payment

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Conversation states ──────────────────────────────────────────────────────
CHECKOUT_OR_MORE = 1
CONFIRM_ORDER = 2
CHOOSE_PAYMENT = 3

# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _ask_checkout_or_more(update, context):
    """Ask user if they want to checkout or add more items."""
    context.user_data["awaiting_checkout_or_more"] = True
    await update.message.reply_text(
        "\U0001f6d2 *Items added to cart!*\n\n"
        "What would you like to do?\n"
        "\u2022 Reply *checkout* to proceed to order\n"
        "\u2022 Reply *more* to add more items",
        parse_mode="Markdown",
    )
    return CHECKOUT_OR_MORE


def _escape_md(text: str) -> str:
    """Escape Markdown special characters in LLM output."""
    for ch in ("_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"):
        text = text.replace(ch, f"\\{ch}")
    return text


async def _send_md(message, text):
    """Send a message with Markdown, falling back to plain text if parsing fails."""
    try:
        await message.reply_text(text, parse_mode="Markdown")
    except Exception:
        # Strip markdown and send as plain text
        plain = text.replace("*", "").replace("_", "")
        await message.reply_text(plain)


async def _preview_and_confirm(update, context):
    """Run order preview check. Only ask to place if all conditions are met."""
    try:
        result = await preview_order()
        breakdown = result["breakdown"]
        total = result["total"]
        delivery = result["delivery_charge"]

        # Check conditions in code
        issues = []
        if total <= 100:
            issues.append(f"Total (Rs.{total:.0f}) is too low — minimum Rs.100")
        if total >= 370:
            issues.append(f"Total (Rs.{total:.0f}) is too high — maximum Rs.370")
        if delivery > 0:
            issues.append(f"Delivery charge Rs.{delivery:.0f} applies — should be free")
        if delivery < 0:
            issues.append("Could not determine delivery charge")

        if not issues:
            context.user_data["awaiting_confirm"] = True
            await _send_md(
                update.message,
                f"\U0001f6d2 Order preview:\n\n{breakdown}\n\n"
                f"Reply *yes* to place the order or *no* to cancel.",
            )
            return CONFIRM_ORDER
        else:
            reasons = "\n".join(f"\u26a0 {i}" for i in issues)
            await _send_md(
                update.message,
                f"\u274c Can't place order:\n{reasons}\n\n{breakdown}\n\n"
                f"Add or remove items and try *checkout* again.",
            )
            return ConversationHandler.END
    except Exception as e:
        logger.error(f"Order preview failed: {e}", exc_info=True)
        await update.message.reply_text(
            f"Sorry, couldn't preview the order: {e}\nPlease try again."
        )
        return ConversationHandler.END


async def _ask_payment_method(update, context):
    """Ask user to choose between online payment and COD."""
    context.user_data["awaiting_payment_method"] = True
    await update.message.reply_text(
        "\U0001f4b3 *How would you like to pay?*\n\n"
        "\u2022 Reply *online* for online payment\n"
        "\u2022 Reply *cod* for Cash on Delivery",
        parse_mode="Markdown",
    )
    return CHOOSE_PAYMENT


async def _place_and_pay(update, context, user_id, payment_method="online"):
    """Place order with chosen payment method. Returns ConversationHandler.END."""
    await update.message.reply_text("\u23f3 Placing your order...")
    try:
        result = await place_order(payment_method)
        context.user_data.pop("awaiting_confirm", None)
        context.user_data.pop("awaiting_payment_method", None)
        context.user_data.pop("cart_cleared", None)

        payment_link = result.get("payment_link", "")
        order_id = result.get("order_id", "")
        message = result.get("message", "")

        save_order(user_id, message)

        if payment_link:
            await _send_md(
                update.message,
                f"\U0001f4b3 Order created! Complete payment here:\n\n"
                f"{payment_link}\n\n"
                f"Order ID: {order_id}\n{message}",
            )
        else:
            await _send_md(
                update.message,
                f"\u2705 Order placed! (COD)\n\nOrder ID: {order_id}\n{message}",
            )
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        await update.message.reply_text(
            f"\u274c Order failed: {e}\nYour cart is still saved \u2014 try again or /cancel."
        )
    return ConversationHandler.END

# ─── Core message handler ─────────────────────────────────────────────────────

async def handle_message(
    user_id: int,
    username: str,
    text: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Central handler — receives plain text regardless of whether it came
    from a typed message or a transcribed voice note.
    """
    logger.info(f"Message from {username} ({user_id}): {text[:80]}")

    # Check if user is replying to checkout-or-more prompt
    if context.user_data.get("awaiting_checkout_or_more"):
        reply = text.lower().strip()
        context.user_data.pop("awaiting_checkout_or_more", None)
        if reply in ("checkout", "check out", "proceed", "done", "place", "order"):
            return await _preview_and_confirm(update, context)
        elif reply in ("more", "add more", "add", "continue"):
            await update.message.reply_text("\U0001f4dd Tell me what items to add.")
            return ConversationHandler.END
        else:
            # Treat as new grocery items
            pass  # fall through to normal handling below

    # Check if user is replying to payment method choice
    if context.user_data.get("awaiting_payment_method"):
        reply = text.lower().strip()
        context.user_data.pop("awaiting_payment_method", None)
        if reply in ("online", "online payment", "pay online", "upi", "card"):
            return await _place_and_pay(update, context, user_id, "online")
        elif reply in ("cod", "cash", "cash on delivery", "pay on delivery"):
            return await _place_and_pay(update, context, user_id, "cod")
        else:
            await update.message.reply_text(
                "Please reply *online* or *cod*.",
                parse_mode="Markdown",
            )
            context.user_data["awaiting_payment_method"] = True
            return CHOOSE_PAYMENT

    # Check if user is replying to order confirmation (yes/no)
    if context.user_data.get("awaiting_confirm"):
        reply = text.lower().strip()
        if reply in ("yes", "y", "confirm", "ok", "place order", "place"):
            return await _ask_payment_method(update, context)
        elif reply in ("no", "n", "cancel", "nope"):
            context.user_data.pop("awaiting_confirm", None)
            await update.message.reply_text(
                "\u274c Order cancelled. Your cart items remain \u2014 send a new request anytime."
            )
            return ConversationHandler.END
        else:
            await update.message.reply_text(
                "Please reply *yes* to place the order or *no* to cancel.",
                parse_mode="Markdown",
            )
            return CONFIRM_ORDER

    # Check if user wants to see their cart
    if text.lower().strip() in ("show cart", "view cart", "my cart", "cart", "what's in my cart"):
        await update.message.reply_text("\U0001f50d Fetching your cart...")
        try:
            status = await get_cart_status()
            await update.message.reply_text(
                f"\U0001f6d2 *Your cart:*\n\n{status}",
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error(f"Cart status failed: {e}")
            await update.message.reply_text("Couldn't fetch cart. Try again later.")
            return ConversationHandler.END
        return await _ask_checkout_or_more(update, context)

    # Check if user is replying with option selections (e.g. "1a, 2b")
    pending_options = context.user_data.get("pending_options")
    pending_items = context.user_data.get("pending_new_items")
    if pending_options and pending_items:
        import re
        codes = [c.strip().lower() for c in re.split(r"[,\s]+", text) if c.strip()]
        matched = [(code, pending_options[code]) for code in codes if code in pending_options]

        if matched:
            # Save picks as product preferences
            # Map option number (1,2,3...) back to item keyword
            for code, product in matched:
                item_num = int(code.rstrip("abcdefgh")) - 1
                if item_num < len(pending_items):
                    item_keyword = pending_items[item_num]
                    # Extract just the product name (before price)
                    product_name = product.split(" - ₹")[0].strip()
                    set_product_pref(user_id, item_keyword, product_name)

            # Clear pending state
            context.user_data.pop("pending_options", None)
            context.user_data.pop("pending_new_items", None)

            # Add selected products to cart
            selections = ", ".join(product.split(" - ₹")[0].strip() for _, product in matched)
            await update.message.reply_text(
                f"\u2705 Saved your picks! Adding to Zepto...\n_{selections}_",
                parse_mode="Markdown",
            )
            try:
                cart_summary = await add_to_cart(selections)
                await update.message.reply_text(
                    f"\U0001f6d2 {cart_summary}", parse_mode="Markdown"
                )
                return await _ask_checkout_or_more(update, context)
            except Exception as e:
                logger.error(f"Add to cart failed: {e}")
                await update.message.reply_text(
                    "Sorry, couldn't add items to Zepto. Please try again."
                )
            return ConversationHandler.END

    member = get_member(user_id)
    if not member:
        await update.message.reply_text(
            "Please /register first before placing orders."
        )
        return

    intent = parse_grocery_intent(text)

    if not intent["is_grocery"]:
        await update.message.reply_text(
            "That doesn't look like a grocery request. Try something like:\n"
            "_\"add milk, eggs and bread\"_ or _\"ingredients for pasta\"_",
            parse_mode="Markdown"
        )
        return

    # Check against household preferences
    household = get_household_members(member["household_id"])
    result    = reconcile_with_preferences(intent["items"], household)

    reply = ""

    if result["flagged"]:
        flags = "\n".join(
            f"\u26a0\ufe0f *{f['item']}* \u2014 conflicts with {f['member']}'s preferences "
            f"({f['reason']})"
            + (f"\n   Substitute: _{f['substitute']}_" if f['substitute'] else "")
            for f in result["flagged"]
        )
        reply += f"*Preference conflicts:*\n{flags}\n\n"

    if result["safe_items"]:
        safe = "\n".join(f"\u2705 {item}" for item in result["safe_items"])
        reply += f"*Ready to add:*\n{safe}"

    await update.message.reply_text(reply, parse_mode="Markdown")

    # Split safe items into known (have stored product pref) vs new (need search)
    if result["safe_items"]:
        # Clear cart only once per session (first grocery request)
        if not context.user_data.get("cart_cleared"):
            await update.message.reply_text("\U0001f9f9 Clearing cart...")
            try:
                await clear_cart()
                context.user_data["cart_cleared"] = True
            except Exception as e:
                logger.error(f"Cart clear failed: {e}")

        known_items = []   # [(item_keyword, zepto_product), ...]
        new_items   = []   # [item_keyword, ...]

        for item in result["safe_items"]:
            pref = get_product_pref(user_id, item)
            if pref:
                known_items.append((item, pref))
            else:
                new_items.append(item)

        # Auto-add known items
        if known_items:
            known_list = "\n".join(
                f"\u2705 {item} \u2192 _{product}_" for item, product in known_items
            )
            await update.message.reply_text(
                f"*Using your saved picks:*\n{known_list}\n\n\U0001f50d Adding to Zepto...",
                parse_mode="Markdown",
            )
            try:
                selections = ", ".join(product for _, product in known_items)
                cart_summary = await add_to_cart(selections)
                await update.message.reply_text(
                    f"\U0001f6d2 {cart_summary}", parse_mode="Markdown"
                )
                if not new_items:
                    return await _ask_checkout_or_more(update, context)
            except Exception as e:
                logger.error(f"Auto-add failed: {e}")
                await update.message.reply_text(
                    "Sorry, couldn't add saved items to Zepto. Please try again."
                )

        # Search for new items — show options
        if new_items:
            await update.message.reply_text(
                f"\U0001f50d Searching Zepto for {len(new_items)} new item(s)..."
            )
            try:
                search_result = await search_zepto(new_items)
                # Store options mapping and pending items for reply handling
                context.user_data["pending_new_items"] = new_items
                context.user_data["pending_options"] = search_result["options"]

                display = search_result["display"]
                if search_result["options"]:
                    await update.message.reply_text(
                        f"*Available options:*\n\n{display}\n\n"
                        f"Reply with option numbers (e.g. _1a, 2b_) to select.\n"
                        f"I'll remember your picks for next time!",
                        parse_mode="Markdown",
                    )
                else:
                    # LLM didn't return structured JSON, show raw
                    await update.message.reply_text(
                        f"*Search results:*\n\n{display}",
                        parse_mode="Markdown",
                    )
            except Exception as e:
                logger.error(f"Zepto search failed: {e}")
                await update.message.reply_text(
                    "Sorry, something went wrong searching Zepto. Please try again."
                )


# ─── Text message handler ─────────────────────────────────────────────────────

async def on_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user     = update.effective_user
    text     = update.message.text or ""
    username = user.first_name or f"User {user.id}"
    return await handle_message(user.id, username, text, update, context)


# ─── Voice note handler ───────────────────────────────────────────────────────

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
        return await handle_message(user.id, username, text, update, context)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─── Commands ─────────────────────────────────────────────────────────────────

async def cmd_register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    args = context.args  # words after /register

    if not args:
        await update.message.reply_text(
            "Usage: /register <your name>\nExample: /register Amar"
        )
        return

    name = " ".join(args).strip()
    existing = get_member(user.id)
    register_member(user.id, name)

    if existing:
        await update.message.reply_text(
            f"\u2705 Updated your name to *{name}*!",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"\u2705 Registered as *{name}*!\n\n"
            f"Next, set your dietary preferences:\n"
            f"/preferences vegetarian, no gluten",
            parse_mode="Markdown"
        )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user     = update.effective_user
    username = user.first_name or "there"
    await update.message.reply_text(
        f"👋 Hey {username}! I'm your household grocery bot.\n\n"
        f"Send me a message like:\n"
        f"  • _\"add milk, eggs and bread\"_\n"
        f"  • _\"ingredients for butter chicken for 4\"_\n"
        f"  • Or send a voice note 🎙\n\n"
        f"First, register yourself:\n"
        f"👉 /register {username}",
        parse_mode="Markdown",
    )


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
        f"\u2705 Preferences saved: _{prefs}_\n\n"
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
    await update.message.reply_text("\u2705 Preferences cleared.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Available commands:*\n\n"
        "/start — Welcome message\n"
        "/register <name> — Join the household\n"
        "/preferences <restrictions> — Set dietary restrictions\n"
        "/myprefs — View your preferences\n"
        "/status — Check your Zepto cart\n"
        "/history — Your last 3 orders\n"
        "/clearprefs — Reset your preferences\n"
        "/help — This message\n\n"
        "Or just type (or voice) what you want to order! 🛒",
        parse_mode="Markdown",
    )


async def cmd_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I don't know that command yet. Try /help to see what I can do."
    )


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    orders = get_recent_orders(user.id, limit=3)
    if not orders:
        await update.message.reply_text("No order history yet.")
        return
    text = "\n\n".join(
        f"\U0001f4e6 *{o['ordered_at'][:10]}*\n{o['summary'][:200]}"
        for o in orders
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("\U0001f50d Checking your Zepto cart...")
    try:
        status = await get_cart_status()
        await update.message.reply_text(
            f"\U0001f6d2 *Your cart:*\n\n{status}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error(f"Cart status failed: {e}")
        await update.message.reply_text("Couldn't fetch cart status. Try again later.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}", exc_info=context.error)


# ─── Cart confirmation handler ────────────────────────────────────────────────

async def checkout_or_more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the checkout/more decision after items are added."""
    reply = update.message.text.lower().strip()

    if reply in ("checkout", "check out", "proceed", "done", "place", "order"):
        return await _preview_and_confirm(update, context)
    elif reply in ("more", "add more", "add", "continue"):
        await update.message.reply_text(
            "\U0001f4dd Tell me what items to add.",
        )
        return ConversationHandler.END
    else:
        # Treat any other text as new grocery items to add
        user = update.effective_user
        username = user.first_name or f"User {user.id}"
        return await handle_message(user.id, username, reply, update, context)


async def confirm_order(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle yes/no after order preview."""
    reply = update.message.text.lower().strip()

    if reply in ("yes", "y", "confirm", "ok", "place order", "place"):
        return await _ask_payment_method(update, context)
    elif reply in ("no", "n", "cancel", "nope"):
        context.user_data.pop("awaiting_confirm", None)
        await update.message.reply_text(
            "\u274c Order cancelled. Your cart items remain — send a new request anytime."
        )
    else:
        await update.message.reply_text(
            "Please reply *yes* to place the order or *no* to cancel.",
            parse_mode="Markdown",
        )
        return CONFIRM_ORDER

    return ConversationHandler.END


async def choose_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle online/cod payment choice."""
    reply = update.message.text.lower().strip()
    user_id = update.effective_user.id

    if reply in ("online", "online payment", "pay online", "upi", "card"):
        return await _place_and_pay(update, context, user_id, "online")
    elif reply in ("cod", "cash", "cash on delivery", "pay on delivery"):
        return await _place_and_pay(update, context, user_id, "cod")
    else:
        await update.message.reply_text(
            "Please reply *online* or *cod*.",
            parse_mode="Markdown",
        )
        return CHOOSE_PAYMENT


async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("awaiting_confirm", None)
    context.user_data.pop("awaiting_checkout_or_more", None)
    context.user_data.pop("awaiting_payment_method", None)
    context.user_data.pop("pending_options", None)
    context.user_data.pop("pending_new_items", None)
    context.user_data.pop("cart_cleared", None)
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# ─── Startup ──────────────────────────────────────────────────────────────────

def main():
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set in .env")
        raise SystemExit(1)

    logger.info("Starting Grocery Bot...")

    request = HTTPXRequest(connection_pool_size=8, read_timeout=30, write_timeout=30, httpx_kwargs={"verify": False})
    get_updates_request = HTTPXRequest(connection_pool_size=1, read_timeout=30, write_timeout=30, httpx_kwargs={"verify": False})
    app = Application.builder().token(token).request(request).get_updates_request(get_updates_request).build()

    init_db()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("register",   cmd_register))
    app.add_handler(CommandHandler("preferences", cmd_preferences))
    app.add_handler(CommandHandler("myprefs",     cmd_myprefs))
    app.add_handler(CommandHandler("clearprefs",  cmd_clearprefs))
    app.add_handler(CommandHandler("history",     cmd_history))
    app.add_handler(CommandHandler("status",      cmd_status))

    # Conversation handler for grocery ordering flow
    conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_message),
            MessageHandler(filters.VOICE, on_voice_message),
        ],
        states={
            CHECKOUT_OR_MORE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, checkout_or_more),
            ],
            CONFIRM_ORDER: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, confirm_order),
            ],
            CHOOSE_PAYMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, choose_payment),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
        allow_reentry=True,
    )
    app.add_handler(conv_handler)

    app.add_handler(MessageHandler(filters.COMMAND, cmd_unknown))
    app.add_error_handler(on_error)

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
