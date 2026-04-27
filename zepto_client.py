# zepto_client.py
import asyncio
import json
import os
import tempfile
import webbrowser
import httpx
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, ImageContent, EmbeddedResource, ResourceLink
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    http_client=httpx.Client(verify=False),
)

MODEL = "openai/gpt-oss-120b:free"

# Cache for widget HTML fetched from MCP resources
_widget_cache: dict[str, str] = {}


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


def render_html(html: str, data: dict | None = None, title: str = "Zepto"):
    """Save HTML to a temp file and open in browser."""
    
    # Inject a stub for list_saved_addresses to prevent 404 errors
    # from widget JS calling the broken endpoint directly
    address_stub = """
    <script>
      // Stub out broken address endpoint calls from widget JS
      const _origFetch = window.fetch;
      window.fetch = function(url, ...args) {
        if (typeof url === 'string' && url.includes('list_saved_addresses')) {
          console.warn('[Zepto Client] Suppressed list_saved_addresses call');
          return Promise.resolve(new Response(
            JSON.stringify({ addresses: [], status: 'ok' }),
            { status: 200, headers: { 'Content-Type': 'application/json' } }
          ));
        }
        return _origFetch(url, ...args);
      };
    </script>
    """
    
    if data:
        inject = f'<script>window.__MCP_DATA__ = {json.dumps(data)};</script>'
        if '</head>' in html:
            html = html.replace('</head>', f'{address_stub}{inject}</head>')
        elif '<body' in html:
            html = html.replace('<body', f'{address_stub}{inject}<body', 1)
        else:
            html = address_stub + inject + html
    else:
        # Still inject the stub even when no data
        if '</head>' in html:
            html = html.replace('</head>', f'{address_stub}</head>')
        elif '<body' in html:
            html = html.replace('<body', f'{address_stub}<body', 1)
        else:
            html = address_stub + html

    tmp = tempfile.NamedTemporaryFile(
        suffix='.html', delete=False, mode='w', encoding='utf-8'
    )
    tmp.write(html)
    tmp.close()
    webbrowser.open(f'file://{tmp.name}')
    print(f"  [🌐 UI opened in browser: {title}]")


async def chat(session, tools, tool_meta, user_message, history):
    history.append({"role": "user", "content": user_message})

    max_retries = 3
    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a helpful Zepto shopping assistant. Use available tools to help "
                        "users search products, manage their cart, place orders, and view order history. "
                        "Keep responses concise and friendly. "
                        "The user's delivery address is already selected at startup. "
                        "Do NOT call list_saved_addresses or select_saved_address unless the user "
                        "explicitly asks to change their delivery address. "
                        "NEVER use Cash on Delivery as a payment method. Always prefer online payment options."
                    )
                }] + history,
                tools=tools,
                max_tokens=800,
            )
        except Exception as e:
            if "tool_use_failed" in str(e) and max_retries > 0:
                max_retries -= 1
                continue
            raise

        msg = response.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            history.append({"role": "assistant", "content": answer})
            return answer

        history.append(msg)

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            # Fix LLM sending JSON strings instead of proper types
            for key, val in fn_args.items():
                if isinstance(val, str):
                    # Try to parse stringified JSON arrays/objects
                    if val.startswith(("[", "{")):
                        try:
                            fn_args[key] = json.loads(val)
                        except json.JSONDecodeError:
                            pass
                    # Fix stringified booleans
                    elif val.lower() == "true":
                        fn_args[key] = True
                    elif val.lower() == "false":
                        fn_args[key] = False

            print(f"  [→ {fn_name}({fn_args})]")

            try:
                result = await session.call_tool(fn_name, fn_args)
            except Exception as call_err:
                print(f"  [⚠ {fn_name} exception: {call_err}]")
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error calling {fn_name}: {call_err}",
                })
                continue

            # Handle tool errors gracefully
            if result.isError:
                err_text = result.content[0].text if result.content else "Tool call failed"
                print(f"  [⚠ {fn_name} error: {err_text}]")
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: {err_text}",
                })
                continue

            if DEBUG:
                print(f"  [DEBUG] structuredContent: {result.structuredContent}")
                print(f"  [DEBUG] content types: {[type(c).__name__ for c in result.content]}")
                for i, c in enumerate(result.content):
                    print(f"  [DEBUG] content[{i}]: type={c.type}, meta={getattr(c, 'meta', None)}")
                    if hasattr(c, 'annotations'):
                        print(f"  [DEBUG]   annotations={c.annotations}")
                    if isinstance(c, ResourceLink):
                        print(f"  [DEBUG]   uri={c.uri}, mimeType={getattr(c, 'mimeType', None)}")
                    if isinstance(c, EmbeddedResource):
                        r = c.resource
                        print(f"  [DEBUG]   resource uri={r.uri}, mime={getattr(r, 'mimeType', None)}, text_len={len(getattr(r, 'text', '') or '')}")

            result_text = ""
            rendered_ui = False

            # 1. Check for structuredContent + widget from _meta
            if result.structuredContent:
                widget_uri = tool_meta.get(fn_name, {}).get("openai/outputTemplate")
                if widget_uri and widget_uri in _widget_cache:
                    render_html(_widget_cache[widget_uri], result.structuredContent, fn_name)
                    rendered_ui = True
                elif widget_uri:
                    # Try to fetch the widget resource
                    try:
                        res = await session.read_resource(widget_uri)
                        for rc in res.contents:
                            html = getattr(rc, 'text', None)
                            if html:
                                _widget_cache[widget_uri] = html
                                render_html(html, result.structuredContent, fn_name)
                                rendered_ui = True
                                break
                    except Exception as e:
                        if DEBUG:
                            print(f"  [DEBUG] Failed to read widget resource {widget_uri}: {e}")

            # 2. Process content blocks
            for item in result.content:
                if isinstance(item, TextContent):
                    result_text += item.text
                elif isinstance(item, ResourceLink):
                    # Resource link in results — try to read and render it
                    mime = getattr(item, 'mimeType', '') or ''
                    if 'html' in mime:
                        try:
                            res = await session.read_resource(str(item.uri))
                            for rc in res.contents:
                                html = getattr(rc, 'text', None)
                                if html:
                                    render_html(html, result.structuredContent, fn_name)
                                    rendered_ui = True
                                    break
                        except Exception as e:
                            if DEBUG:
                                print(f"  [DEBUG] Failed to read resource link {item.uri}: {e}")
                elif isinstance(item, EmbeddedResource):
                    resource = item.resource
                    mime = getattr(resource, 'mimeType', '') or ''
                    text = getattr(resource, 'text', '') or ''
                    if 'html' in mime and text:
                        render_html(text, result.structuredContent, fn_name)
                        rendered_ui = True
                    elif text:
                        result_text += text

            # 3. If we got structuredContent but no widget, render it as formatted JSON in HTML
            if result.structuredContent and not rendered_ui:
                sc = result.structuredContent
                html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Zepto - {fn_name}</title>
<style>body{{font-family:system-ui;margin:2rem;background:#f8f9fa}}
pre{{background:#fff;padding:1rem;border-radius:8px;border:1px solid #dee2e6;overflow-x:auto}}
h2{{color:#7b2d8e}}</style></head>
<body><h2>🛒 {fn_name}</h2><pre>{json.dumps(sc, indent=2, ensure_ascii=False)}</pre></body></html>"""
                render_html(html, title=fn_name)
                rendered_ui = True

            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text or json.dumps(result.structuredContent) if result.structuredContent else result_text or "Done",
            })


async def main():
    server_params = StdioServerParameters(
        command="npx",
        args=["mcp-remote", "https://mcp.zepto.co.in/mcp"],
        env={**os.environ, "NODE_TLS_REJECT_UNAUTHORIZED": "0"},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            # Filter out broken tools (none currently)
            BROKEN_TOOLS: set[str] = set()
            mcp_tools.tools = [t for t in mcp_tools.tools if t.name not in BROKEN_TOOLS]
            tools = to_openai_tools(mcp_tools)
            print(f"✅ Connected ({len(tools)} tools loaded)")
            for t in mcp_tools.tools:
                print(f"   • {t.name}")
            print()

            # Extract _meta from tool definitions (e.g. openai/outputTemplate)
            tool_meta = {}
            for t in mcp_tools.tools:
                if t.meta:
                    tool_meta[t.name] = t.meta
                    if DEBUG:
                        print(f"  [DEBUG] Tool {t.name} _meta: {t.meta}")

            # Discover MCP resources (widget HTML)
            try:
                resources = await session.list_resources()
                if resources.resources:
                    print(f"📦 {len(resources.resources)} resources found:")
                    for r in resources.resources:
                        mime = getattr(r, 'mimeType', '') or ''
                        print(f"   • {r.uri} ({mime})")
                        # Pre-cache HTML widget resources
                        if 'html' in mime:
                            try:
                                res = await session.read_resource(str(r.uri))
                                for rc in res.contents:
                                    html = getattr(rc, 'text', None)
                                    if html:
                                        _widget_cache[str(r.uri)] = html
                                        if DEBUG:
                                            print(f"  [DEBUG] Cached widget: {r.uri} ({len(html)} chars)")
                            except Exception as e:
                                if DEBUG:
                                    print(f"  [DEBUG] Could not read resource {r.uri}: {e}")
                    print()
            except Exception as e:
                if DEBUG:
                    print(f"  [DEBUG] list_resources failed: {e}")

            history = []

            # Boot: load profile
            print("Loading your Zepto profile...\n")
            boot_response = await chat(
                session, tools, tool_meta,
                "Load my Zepto profile using get_user_details. "
                "Then call list_saved_addresses to get my addresses, and use "
                "select_saved_address to select the one labelled 'Home'. "
                "Also check my current cart with view_cart. "
                "If any tool returns an error, just skip it and report what worked.",
                history
            )
            print(f"Assistant: {boot_response}\n")

            print("─" * 40)
            print("Session ready. Type your grocery requests below.")
            print("Type 'quit' to exit.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input or user_input.lower() in ("quit", "exit", "q"):
                    break

                response = await chat(session, tools, tool_meta, user_input, history)
                print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())