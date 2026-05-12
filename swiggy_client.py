# swiggy_client.py
#
# Standalone script to query the Swiggy Instamart MCP server.
# Uses npx mcp-remote (stdio transport) — same as zepto_client.py.
#
# HOW TO RUN:
#   pip install mcp groq python-dotenv
#   python swiggy_client.py
#
# NOTE: First run will open a browser for Swiggy OAuth authentication.

import asyncio
import json
import os
import sys
import logging

import httpx
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.WARNING,  # Keep logs quiet — only show errors
)

# ─── Config ──────────────────────────────────────────────────────────────────

SWIGGY_MCP_URL = os.environ.get("SWIGGY_MCP_URL", "https://mcp.swiggy.com/im")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "openai/gpt-oss-120b:free"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def to_openai_tools(mcp_tools) -> list:
    """Convert MCP tool definitions to OpenAI-compatible format for Groq."""
    tools = []
    for t in mcp_tools.tools:
        tools.append({
            "type": "function",
            "function": {
                "name":        t.name,
                "description": t.description or "",
                "parameters":  t.inputSchema,
            }
        })
    return tools


def print_tools(tools: list):
    """Pretty-print available tools."""
    print(f"\n{'─'*50}")
    print(f"  {len(tools)} tools available on Swiggy Instamart MCP")
    print(f"{'─'*50}")
    for t in tools:
        desc = t["function"]["description"].split("\n")[0][:60]
        print(f"  • {t['function']['name']:<30} {desc}")
    print(f"{'─'*50}\n")


# ─── Core chat loop ───────────────────────────────────────────────────────────

async def chat(session: ClientSession, tools: list, history: list, user_input: str) -> str:
    """
    Single turn of the LLM + tool calling loop.
    Returns the final text response after all tool calls are resolved.
    """
    if not OPENROUTER_API_KEY:
        # No LLM — call tools directly based on keywords
        return await direct_tool_call(session, user_input)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        http_client=httpx.Client(verify=False),
    )

    system_prompt = """You are a helpful Swiggy Instamart grocery assistant.
Help the user search for products, manage their cart, and place orders.
Always call get_addresses first before searching for products.
Be concise and practical in your responses."""

    history.append({"role": "user", "content": user_input})

    for _ in range(15):  # Max 15 tool calls per turn
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + history,
            tools=tools,
            max_tokens=1000,
        )

        msg = response.choices[0].message

        # No more tool calls — return final text response
        if not msg.tool_calls:
            reply = msg.content or "Done."
            history.append({"role": "assistant", "content": reply})
            return reply

        # Process tool calls
        history.append(msg)

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args   = json.loads(tc.function.arguments)
                print(f"  ⚙ calling {tool_name}({json.dumps(args, ensure_ascii=False)[:80]}...)")
                result = await session.call_tool(tool_name, args)
                content = "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                )
                history.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      content or "{}",
                })
            except Exception as e:
                print(f"  ✗ {tool_name} error: {e}")
                history.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      f"Error: {e}",
                })

    return "Reached max tool calls."


async def direct_tool_call(session: ClientSession, user_input: str) -> str:
    """
    Fallback when no GROQ_API_KEY — call tools directly based on keywords.
    Useful for quick testing without burning API quota.
    """
    text = user_input.lower().strip()

    if text in ("tools", "list tools", "help"):
        tools = await session.list_tools()
        lines = [f"  • {t.name}: {(t.description or '')[:60]}" for t in tools.tools]
        return "\n".join(lines)

    if text in ("addresses", "my address", "get addresses"):
        result = await session.call_tool("get_addresses", {})
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    if text in ("cart", "my cart", "show cart"):
        result = await session.call_tool("get_cart", {})
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    if text in ("orders", "my orders", "order history"):
        result = await session.call_tool("get_orders", {"orderType": "INSTAMART", "count": 5})
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    if text == "clear cart":
        result = await session.call_tool("clear_cart", {})
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    return (
        "No GROQ_API_KEY set — only direct commands work:\n"
        "  addresses, cart, orders, clear cart, tools"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("\n🛒 Swiggy Instamart MCP Client")
    print("   Connecting via mcp-remote...")

    server_params = StdioServerParameters(
        command="npx",
        args=["mcp-remote", SWIGGY_MCP_URL],
        env={**os.environ, "NODE_TLS_REJECT_UNAUTHORIZED": "0"},
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                mcp_tools = await session.list_tools()
                oa_tools  = to_openai_tools(mcp_tools)

                print_tools(oa_tools)

                if not OPENROUTER_API_KEY:
                    print("  ⚠ OPENROUTER_API_KEY not set — running in direct tool mode")
                    print("  Commands: addresses, cart, orders, clear cart, tools\n")
                else:
                    print("  ✓ OpenRouter connected — full natural language mode\n")

                history = []

                print("Type your request or 'quit' to exit.\n")

                while True:
                    try:
                        user_input = input("You: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        print("\nGoodbye!")
                        break

                    if not user_input:
                        continue

                    if user_input.lower() in ("quit", "exit", "q"):
                        print("Goodbye!")
                        break

                    # Special commands
                    if user_input.lower() == "clear history":
                        history = []
                        print("  Conversation history cleared.\n")
                        continue

                    if user_input.lower() == "tools":
                        print_tools(oa_tools)
                        continue

                    print()
                    response = await chat(session, oa_tools, history, user_input)
                    print(f"Bot: {response}\n")

    except Exception as e:
        import traceback
        print(f"\n✗ Connection failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
