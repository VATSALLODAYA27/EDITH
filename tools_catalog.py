"""Print every agent's tools exactly as the LLM sees them (generated from the code, so it's never out of date).

    python tools_catalog.py              -> one line per tool: agent | tool(args) | description
    python tools_catalog.py read_sheet   -> the full JSON schema sent to the LLM for that tool
"""
import json
import sys

from langchain_core.utils.function_calling import convert_to_openai_tool

from agents import browser, calendar_agent, document, editor, excel, mail, ppt

AGENT_TOOLS = {
    "document_agent": document.TOOLS, "document_editor": editor.TOOLS, "excel_agent": excel.TOOLS, "ppt_agent": ppt.TOOLS,
    "browser_agent": browser.TOOLS, "email_agent": mail.TOOLS, "calendar_agent": calendar_agent.TOOLS,
}
# rag_agent and research_agent have no tools: RAG always retrieves (a fixed pipeline), research just answers.

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1:
        tool = next(t for tools in AGENT_TOOLS.values() for t in tools if t.name == sys.argv[1])
        print(json.dumps(convert_to_openai_tool(tool), indent=2))
    else:
        total = 0
        for agent, tools in AGENT_TOOLS.items():
            for t in tools:
                args = ", ".join(f"{k}{'' if k in t.tool_call_schema.model_json_schema().get('required', []) else '?'}"
                                 for k in t.args)
                print(f"{agent:<15} {t.name}({args})  -  {t.description.splitlines()[0]}")
                total += 1
        print(f"\n{total} tools across {len(AGENT_TOOLS)} agents  (? = optional argument)")
