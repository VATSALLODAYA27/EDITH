"""Phase 4 tests: multi-agent workflows (sequential chains + parallel steps) through the Orchestrator.
    
Each test checks ORDER (which turn each agent ran in), DATA FLOW (inputs) and the resulting STATE
(files, drafts), not just the answer text.   Run:  python -m tests.workflows_test
"""
import os
import tempfile

# Use a throwaway memory database: tests must never write into the user's real history/profile.
_tmp = tempfile.mkdtemp()
os.environ["MEMORY_DB"] = os.path.join(_tmp, "test_memory.sqlite")
os.environ["TRACE_LOG"] = os.path.join(_tmp, "trace.jsonl")  # ...and never into the real trace log
import sys
from datetime import date, timedelta

from openpyxl import load_workbook

from agents.calendar_agent import reset_calendar
from agents.document import read_document
from agents.mail import _load as load_mailbox, reset_mailbox
from agents.ppt import read_presentation
from orchestrator import run
from userdata import use_temp_data, workspace_dir  # noqa: E402

use_temp_data()  # all per-user files (workspace, mailbox, calendar) go to a temp folder, never the real ones
WORKSPACE = workspace_dir()

sys.stdout.reconfigure(encoding="utf-8")


def turn_of(result, agent):
    return next(h["turn"] for h in result["history"] if h["agent"] == agent)


def inputs_of(result, agent):
    return next(h["inputs"] for h in result["history"] if h["agent"] == agent)


def fresh(*files):
    for f in files:
        (WORKSPACE / f).unlink(missing_ok=True)


def check(name, question, fn):
    print(f"\n=== {name}: {question}")
    result = run(question)
    print(f"history: {[(h['turn'], h['agent'], h['inputs']) for h in result['history']]}")
    calls = [h["agent"] for h in result["history"]]
    assert len(calls) == len(set(calls)), f"an agent ran twice (wasted work / loop): {calls}"
    before = {f.name for f in WORKSPACE.iterdir()}
    fn(result)
    print(f"--> {name} OK")


reset_mailbox()
reset_calendar()


# 1. PARALLEL: two independent questions -> both agents in the SAME turn
def parallel(r):
    assert {"rag_agent", "research_agent"} <= set(r["agent_results"])
    assert turn_of(r, "rag_agent") == turn_of(r, "research_agent"), "independent steps should run in parallel"
    assert "1,500" in r["final_answer"] and "canberra" in r["final_answer"].lower(), r["final_answer"]


check("parallel", "What's our daily meal allowance when travelling, and what's the capital of Australia?", parallel)


# 2. Document -> PPT: slides built from the report's facts
fresh("survey_deck.pptx")


def doc_to_ppt(r):
    assert turn_of(r, "document_agent") < turn_of(r, "ppt_agent"), "PPT must wait for the document"
    assert "document_agent" in inputs_of(r, "ppt_agent")
    deck = read_presentation.invoke({"filename": "survey_deck.pptx"})
    print(deck)
    assert "82" in deck, "the deck should contain the report's satisfaction figure"


check("doc->ppt", "Read survey_report.pdf and create survey_deck.pptx summarizing its findings in 3 content slides.",
      doc_to_ppt)


# 3. Calendar -> Email: check availability, then draft (not send) the email
def cal_to_email(r):
    assert turn_of(r, "calendar_agent") < turn_of(r, "email_agent")
    assert "calendar_agent" in inputs_of(r, "email_agent")
    box = load_mailbox()
    draft = next(d for d in box["drafts"] if d["to"] == "john.miller@nimbuslabs.com")
    print("draft:", draft["subject"], "|", draft["body"][:120])
    assert "3" in draft["body"] and not box["sent"], draft


check("calendar->email", "Check if I'm free tomorrow at 3 PM for 30 minutes. If I am, draft an email to "
      "john.miller@nimbuslabs.com proposing our Q4 roadmap sync at that time.", cal_to_email)


# 4. RAG -> Document: company facts into a Word file
fresh("policy_summary.docx")


def rag_to_doc(r):
    assert turn_of(r, "rag_agent") < turn_of(r, "document_agent")
    assert "rag_agent" in inputs_of(r, "document_agent")
    text = read_document.invoke({"filename": "policy_summary.docx"})
    assert "24" in text and "3" in text, text  # 24 vacation days, 3 remote days per week


check("rag->document", "Create policy_summary.docx summarizing our vacation and remote-work policies.", rag_to_doc)


# 5. Browser -> Excel: live web fact into a spreadsheet
fresh("versions.xlsx")


def browser_to_excel(r):
    assert turn_of(r, "browser_agent") < turn_of(r, "excel_agent")
    cells = [str(c.value) for row in load_workbook(WORKSPACE / "versions.xlsx").active.iter_rows() for c in row]
    print("cells:", cells)
    assert any(c.startswith("3.1") for c in cells), cells


check("browser->excel", "Find the latest stable version of Python on the web and record it in versions.xlsx "
      "with columns Language and Version.", browser_to_excel)


# 6. RAG -> Excel -> PPT: a 3-agent chain
fresh("expense_limits.xlsx", "expense_limits.pptx")


def three_chain(r):
    t_rag, t_xl, t_ppt = turn_of(r, "rag_agent"), turn_of(r, "excel_agent"), turn_of(r, "ppt_agent")
    assert t_rag < t_xl and t_rag < t_ppt, "both files need the policy facts first"
    assert (WORKSPACE / "expense_limits.xlsx").exists() and (WORKSPACE / "expense_limits.pptx").exists()
    cells = " ".join(str(c.value) for row in load_workbook(WORKSPACE / "expense_limits.xlsx").active.iter_rows()
                     for c in row)
    assert "1500" in cells.replace(",", "") and "7000" in cells.replace(",", ""), cells


check("rag->excel->ppt", "Look up our travel expense limits, put each limit (item and amount) in "
      "expense_limits.xlsx, then create expense_limits.pptx presenting those limits.", three_chain)

reset_mailbox()
reset_calendar()
print("\nWorkflows OK")
