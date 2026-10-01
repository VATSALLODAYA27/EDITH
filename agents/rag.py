"""RAG Agent: answers questions from the company knowledge base and the documents the user uploaded.

    task -> embed -> ChromaDB finds the TOP_K closest chunks -> LLM answers ONLY from those -> result

Two kinds of chunks share one index, told apart by the `owner` metadata:
  "shared"  - company docs in data/docs (everyone)       ids "hr_policy.md#0"
  <user id> - files that user uploaded (only that user)   ids "u:<user>:report.pdf#0"

Run the standalone test from the project root:  python -m agents.rag
"""
from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from agents.document import _chunks, _extract_text
from llm import get_llm
from userdata import CURRENT_USER

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
TOP_K = 3  # how many chunks the LLM gets; more = more context but more noise and tokens
RAG_TYPES = (".pdf", ".docx", ".txt", ".md")  # uploads with extractable text get indexed
UPLOAD_CHUNK = 1500  # characters; uploaded files have no "## " structure, so split by size at paragraph breaks

# Embeddings via the Gemini API (GOOGLE_API_KEY) - no local model. Note: chunk text is sent to Google.
# Vectors from different models are NOT comparable, so the index must be rebuilt if this model changes.
EMBED_MODEL = "models/gemini-embedding-2"


@register_embedding_function
class GeminiEmbeddings(EmbeddingFunction[Documents]):
    """Chroma calls this whenever it needs vectors, so the collection can never fall back to its local default.

    Why not Chroma's built-in GoogleGenaiEmbeddingFunction? With gemini-embedding-2 it returned ONE vector
    for a batch of texts (a multimodal model merges the list into one input). LangChain's wrapper is correct.
    """

    def __init__(self, model: str = EMBED_MODEL):
        self.model = model
        self._lc = GoogleGenerativeAIEmbeddings(model=model)

    def __call__(self, input: Documents) -> Embeddings:  # used when adding documents
        return self._lc.embed_documents(list(input))

    def embed_query(self, input: Documents) -> Embeddings:  # used for query_texts (query-optimised vectors)
        return [self._lc.embed_query(text) for text in input]

    @staticmethod
    def name() -> str:
        return "langchain_gemini"

    def get_config(self) -> dict:  # saved with the collection, so it records which model built the index
        return {"model": self.model}

    @staticmethod
    def build_from_config(config: dict) -> "GeminiEmbeddings":
        return GeminiEmbeddings(config["model"])


# PersistentClient saves the index to disk, so it survives restarts.
client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
collection = client.get_or_create_collection(
    "private_docs_gemini", embedding_function=GeminiEmbeddings(), configuration={"hnsw": {"space": "cosine"}}
)
llm = get_llm()


def chunk(text: str) -> list[str]:
    """One chunk per '## ' section; each chunk keeps the document title for context."""
    # ponytail: heading-based split fits our markdown docs; long unstructured text/PDFs need a
    # size-based splitter (e.g. langchain's RecursiveCharacterTextSplitter) — Document Agent phase.
    title, *sections = text.split("\n## ")
    return [f"{title.strip()}\n## {s.strip()}" for s in sections] or [title.strip()]


def _store(ids: list[str], docs: list[str], metas: list[dict]) -> int:
    """Upsert chunks, embedding only new or changed text (embedding costs API quota). Returns how many were embedded."""
    if not ids:
        return 0
    stored = collection.get(ids=ids, include=["documents", "metadatas"])
    old = {cid: (d, m) for cid, d, m in zip(stored["ids"], stored["documents"], stored["metadatas"])}
    changed = [k for k, cid in enumerate(ids) if cid not in old or old[cid][0] != docs[k]]
    relabel = [k for k, cid in enumerate(ids) if k not in changed and old[cid][1] != metas[k]]
    if changed:
        collection.upsert(  # upsert = insert or update by id
            ids=[ids[k] for k in changed],
            documents=[docs[k] for k in changed],
            metadatas=[metas[k] for k in changed],
        )  # no embeddings= -> Chroma calls GeminiEmbeddings for us
        print(f"[rag] embedded {len(changed)} new/changed chunks via {EMBED_MODEL}")
    if relabel:  # same text, new metadata (e.g. the owner field added): no re-embedding needed
        collection.update(ids=[ids[k] for k in relabel], metadatas=[metas[k] for k in relabel])
    return len(changed)


def ingest() -> int:
    """(Re)index every .md/.txt file in data/docs. Safe to run repeatedly. Returns how many chunks were embedded."""
    ids, docs, metas = [], [], []
    for path in sorted([*DOCS_DIR.glob("*.md"), *DOCS_DIR.glob("*.txt")]):
        for i, text in enumerate(chunk(path.read_text(encoding="utf-8"))):
            ids.append(f"{path.name}#{i}")
            docs.append(text)
            metas.append({"source": path.name, "owner": "shared"})
    embedded = _store(ids, docs, metas)
    # chunks of deleted/shortened shared docs (users' uploads, ids "u:...", are managed by index_file)
    stale = [i for i in collection.get()["ids"] if not i.startswith("u:") and i not in ids]
    if stale:
        collection.delete(ids=stale)
    return embedded


def index_file(filename: str) -> int:
    """Index one of the CURRENT user's workspace files, so their RAG questions can use it (nobody else's can).
    Returns the number of chunks (0 = no extractable text, e.g. a scanned PDF)."""
    owner = CURRENT_USER.get()
    parts = [p.strip() for p in _chunks(_extract_text(filename), UPLOAD_CHUNK) if p.strip()]
    ids = [f"u:{owner}:{filename}#{i}" for i in range(len(parts))]
    _store(ids, [f"{filename}\n{p}" for p in parts], [{"source": filename, "owner": owner}] * len(parts))
    old = collection.get(where={"$and": [{"owner": owner}, {"source": filename}]})["ids"]
    stale = [i for i in old if i not in ids]  # the file got shorter
    if stale:
        collection.delete(ids=stale)
    return len(parts)


def search(query: str, k: int = TOP_K) -> list[dict]:
    """Return the k chunks whose meaning is closest to the query (distance: 0 = identical)."""
    r = collection.query(query_texts=[query], n_results=k,  # Chroma -> GeminiEmbeddings.embed_query
                         where={"owner": {"$in": ["shared", CURRENT_USER.get()]}})  # company docs + MY uploads
    return [
        {"source": m["source"], "text": d, "distance": round(dist, 3)}
        for m, d, dist in zip(r["metadatas"][0], r["documents"][0], r["distances"][0])
    ]


RAG_PROMPT = """You answer questions using ONLY the document excerpts below.
Cite the source file name for every fact, like (hr_policy.md).
If the excerpts don't contain the answer, reply exactly: Not found in documents.

{context}"""


def rag_agent(state: dict) -> dict:
    hits = search(state["task"])
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    answer = llm.invoke([SystemMessage(RAG_PROMPT.format(context=context)), HumanMessage(state["task"])]).text
    print(f"[rag_agent] retrieved {[h['source'] for h in hits]} -> {answer[:80]}...")
    return {"agent_results": {"rag_agent": answer}}


ingest()  # keep the index in sync with data/docs every time the agent is loaded


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Indexed {collection.count()} chunks from {DOCS_DIR}")
    assert ingest() == 0, "nothing changed, so nothing should be re-embedded (wastes API quota)"
    dims = len(collection.get(limit=1, include=["embeddings"])["embeddings"][0])
    assert dims == 3072, f"expected Gemini vectors (3072 dims), got {dims}"  # local MiniLM would be 384
    ef = collection.configuration_json["embedding_function"]
    assert ef["name"] == "langchain_gemini", f"collection is configured with a different embedder: {ef}"

    # 1. Retrieval alone (no LLM). "days off" never appears in the docs -> embeddings match MEANING, not words.
    for query, expected in [("How many days off do I get?", "hr_policy.md"),
                            ("Can I fly business class?", "expense_policy.md"),
                            ("What database does Atlas use?", "project_atlas.md")]:
        hits = search(query)
        print(f"\nsearch: {query}")
        for h in hits:
            print(f"  {h['distance']:.3f}  {h['source']:<20} {h['text'].splitlines()[1]}")
        assert hits[0]["source"] == expected, f"expected {expected} first"

    # 2. The full agent: retrieval + LLM
    print()
    ans = rag_agent({"task": "What is the hotel limit per night in metro cities?"})["agent_results"]["rag_agent"]
    assert "7,000" in ans or "7000" in ans, ans

    # 3. Not in the docs -> must say so instead of making something up
    ans = rag_agent({"task": "What is the CEO's salary?"})["agent_results"]["rag_agent"]
    assert "Not found in documents" in ans, ans

    print("\nRAG agent OK")
