# zepto.py — Zepto MCP cart builder (lifted from zepto_client.py)

import asyncio
import json
import os
import logging
import httpx
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    http_client=httpx.Client(verify=False),
)

MODEL = "openai/gpt-oss-120b:free"
ZEPTO_MCP_URL = os.environ.get("ZEPTO_MCP_URL", "https://mcp.zepto.co.in/mcp")


def _to_openai_tools(mcp_tools) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools.tools
    ]


async def _run_mcp_chat(system_prompt: str, user_message: str, max_rounds: int = 20) -> str:
    """Open an MCP session and run a multi-turn LLM+tool loop. Returns final LLM text."""
    server_params = StdioServerParameters(
        command="npx",
        args=["mcp-remote", ZEPTO_MCP_URL],
        env={**os.environ, "NODE_TLS_REJECT_UNAUTHORIZED": "0"},
    )

    try:
        async with asyncio.timeout(120):  # 2 min hard timeout for entire operation
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    mcp_tools = await session.list_tools()
                    oa_tools = _to_openai_tools(mcp_tools)
                    logger.info(f"Zepto MCP connected ({len(oa_tools)} tools)")

                    history = [{"role": "user", "content": user_message}]

                    for _ in range(max_rounds):
                        # LLM call with retry
                        llm_response = None
                        for attempt in range(3):
                            try:
                                llm_response = client.chat.completions.create(
                                    model=MODEL,
                                    messages=[{"role": "system", "content": system_prompt}] + history,
                                    tools=oa_tools,
                                    max_tokens=1200,
                                )
                                break
                            except Exception as e:
                                err = str(e)
                                if attempt < 2 and ("429" in err or "503" in err or "capacity" in err or "No backends" in err):
                                    logger.warning(f"LLM retry {attempt+1}: {err[:80]}")
                                    await asyncio.sleep(3)
                                else:
                                    raise

                        if not llm_response:
                            return "LLM unavailable — please try again in a moment."

                        msg = llm_response.choices[0].message

                        if not msg.tool_calls:
                            return msg.content or "Done."

                        history.append(msg)

                        for tc in msg.tool_calls:
                            fn_name = tc.function.name
                            try:
                                fn_args = json.loads(tc.function.arguments) or {}
                            except (json.JSONDecodeError, TypeError):
                                fn_args = {}

                            logger.info(f"  [→ {fn_name}({fn_args})]")

                            try:
                                result = await session.call_tool(fn_name, fn_args)
                                content = " ".join(
                                    c.text for c in result.content if isinstance(c, TextContent)
                                )
                                history.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": content or "Done",
                                })
                            except Exception as e:
                                logger.error(f"Tool call error ({fn_name}): {e}")
                                history.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": f"Error: {e}",
                                })

    except TimeoutError:
        logger.error("Zepto MCP operation timed out (120s)")
        return "Zepto is slow right now — please try again in a moment."
    except Exception as e:
        logger.error(f"MCP session error: {e}")
        return f"Zepto connection error: {e}"

    return "Done."


async def search_zepto(items: list[str]) -> dict:
    """
    Search Zepto for each item and return structured results.
    Returns: {
        "display": str,          # formatted text to show user
        "options": {             # mapping of option codes to product names
            "1a": "Amul Toned Milk 1L - ₹32",
            "1b": "Mother Dairy Full Cream 1L - ₹68",
            ...
        }
    }
    """
    system_prompt = (
        "You are a grocery shopping assistant for Zepto. "
        "First call list_saved_addresses and select_saved_address to select the address labelled 'Home'. "
        "Then for each item the user wants, call search_products to find it on Zepto. "
        "After searching ALL items, respond with ONLY a JSON object (no markdown, no code fences) in this exact format:\n"
        '{"options": {"1a": "Product Name - Qty - ₹Price", "1b": "...", "2a": "...", ...}}\n'
        "Number items 1, 2, 3... and options a, b, c within each item. "
        "Include the top 3-5 available options per item with name, quantity/weight, and price. "
        "Do NOT add anything to the cart."
    )

    items_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    user_message = f"Search Zepto for these items and return the options as JSON:\n{items_text}"

    raw = await _run_mcp_chat(system_prompt, user_message)

    # Parse LLM response into structured options
    try:
        # Strip code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        options = data.get("options", {})
    except (json.JSONDecodeError, AttributeError):
        # Fallback: return raw text as display, no structured options
        return {"display": raw, "options": {}}

    # Build display text from options
    lines = []
    current_num = ""
    for code in sorted(options.keys()):
        num = code.rstrip("abcdefgh")
        if num != current_num:
            if current_num:
                lines.append("")
            current_num = num
        lines.append(f"  *{code}*: {options[code]}")

    display = "\n".join(lines)
    return {"display": display, "options": options}


async def clear_cart() -> str:
    """Clear all items from the Zepto cart."""
    system_prompt = (
        "You are a grocery shopping assistant for Zepto. "
        "First call list_saved_addresses and select_saved_address to select the address labelled 'Home'. "
        "Then call view_cart to see current items, then remove all of them by setting each item's quantity to 0. "
        "Confirm once the cart is empty."
    )
    return await _run_mcp_chat(system_prompt, "Clear my cart completely.")


async def add_to_cart(selections: str) -> str:
    """
    Add specific products to the Zepto cart based on user's selections.
    `selections` is a free-text description of what to add (e.g. "add Amul Toned Milk 1L and Britannia bread").
    Returns a cart summary.
    """
    system_prompt = (
        "You are a grocery shopping assistant for Zepto. "        "First call list_saved_addresses and select_saved_address to select the address labelled 'Home'. "        "The user has already seen search results and chosen specific products. "
        "Search for each chosen product, find the exact match, and add it to the cart. "
        "After adding all items, call view_cart to show the final cart summary. "
        "Be precise — add exactly what the user asked for."
    )

    user_message = f"Add these specific products to my cart:\n{selections}"

    return await _run_mcp_chat(system_prompt, user_message)


async def preview_order() -> dict:
    """
    Dry-run create_online_payment_order to get full cost breakdown.
    Returns: {"breakdown": str, "total": float, "delivery_charge": float}
    """
    system_prompt = (
        "You are a grocery shopping assistant for Zepto. "
        "First call list_saved_addresses and select_saved_address to select the address labelled 'Home'. "
        "Then call create_online_payment_order with confirmOrder=False to get an order preview. "
        "Respond with ONLY a JSON object (no markdown, no code fences) in this exact format:\n"
        '{"breakdown": "Item1 x1 - Rs.50\\nItem2 x2 - Rs.80\\n---\\nItem Total: Rs.130\\nDiscount: -Rs.10\\n'
        'Delivery: Rs.0\\nHandling: Rs.5\\nGrand Total: Rs.125",'
        '"total": 125.0, "delivery_charge": 0.0}\n\n'
        "Include every item with quantity and price, then subtotal, discounts, delivery charge, "
        "handling/packaging charge, and grand total. Use the actual values from the preview response."
    )
    raw = await _run_mcp_chat(system_prompt, "Preview my current order with full cost breakdown.")

    # Parse JSON from LLM response
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        return {
            "breakdown": data.get("breakdown", raw),
            "total": float(data.get("total", 0)),
            "delivery_charge": float(data.get("delivery_charge", -1)),
        }
    except (json.JSONDecodeError, ValueError, AttributeError):
        logger.warning(f"Could not parse preview JSON, raw: {raw[:200]}")
        return {"breakdown": raw, "total": 0, "delivery_charge": -1}


async def place_order(payment_method: str = "online") -> dict:
    """
    Place the current Zepto cart order.
    Args:
        payment_method: "online" for online payment, "cod" for cash on delivery.
    Returns: {"order_id": str, "payment_link": str, "message": str}
    """
    if payment_method == "cod":
        system_prompt = (
            "You are a grocery shopping assistant for Zepto. "
            "First call list_saved_addresses and select_saved_address to select the address labelled 'Home'. "
            "Then place the order using Cash on Delivery (COD) payment method. "
            "Look for a tool that supports COD or cash-on-delivery orders. "
            "If only create_online_payment_order is available, call it with confirmOrder=True and paymentMethod='COD' or similar COD option. "
            "Respond with ONLY a JSON object (no markdown, no code fences) in this exact format:\n"
            '{"order_id": "the_order_id", "payment_link": "", "message": "brief summary"}\n'
            "Extract the order ID from the response."
        )
        user_message = "Place my current cart order. Use Cash on Delivery (COD) payment."
    else:
        system_prompt = (
            "You are a grocery shopping assistant for Zepto. "
            "First call list_saved_addresses and select_saved_address to select the address labelled 'Home'. "
            "Then call create_online_payment_order with confirmOrder=True to place the order. "
            "Respond with ONLY a JSON object (no markdown, no code fences) in this exact format:\n"
            '{"order_id": "the_order_id", "payment_link": "https://...", "message": "brief summary"}\n'
            "Extract the order ID and payment link from the response. "
            "If there is no payment link, set payment_link to empty string."
        )
        user_message = "Place my current cart order. Use online payment."

    raw = await _run_mcp_chat(system_prompt, user_message)

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        return {
            "order_id": data.get("order_id", ""),
            "payment_link": data.get("payment_link", ""),
            "message": data.get("message", raw),
        }
    except (json.JSONDecodeError, ValueError, AttributeError):
        logger.warning(f"Could not parse place_order JSON, raw: {raw[:200]}")
        return {"order_id": "", "payment_link": "", "message": raw}


async def check_payment(order_id: str) -> str:
    """Check payment status for an order."""
    system_prompt = (
        "You are a grocery shopping assistant for Zepto. "
        "Call check_payment_status with the given order ID to check if payment is complete. "
        "Report the payment status clearly: PAID, PENDING, or FAILED."
    )
    return await _run_mcp_chat(system_prompt, f"Check payment status for order {order_id}.")


async def get_cart_status() -> str:
    """Get current Zepto cart contents."""
    system_prompt = (
        "You are a grocery shopping assistant for Zepto. "
        "First call list_saved_addresses and select_saved_address to select the address labelled 'Home'. "
        "Then call view_cart to see what's in the user's cart. "
        "Report the items, quantities, and total price in a concise format. "
        "If the cart is empty, say 'Cart is empty'."
    )
    return await _run_mcp_chat(system_prompt, "Show me what's in my cart.")
