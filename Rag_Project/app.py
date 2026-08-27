"""
CourseMate-AI — Streamlit UI for the LangGraph college assistant.

WHAT WAS PRESERVED (unchanged from the original script):
  - State schema (programme / messages / classify / retrieved_context)
  - classifier_node prompt & logic (word-for-word)
  - academic_rag_node / fees_rag_node retrieval logic (same splitter,
    same FAISS + MMR retriever kwargs: k=4, fetch_k=15, lambda_mult=0.25,
    chunk_size=800, chunk_overlap=140)
  - genral_node / response_node prompts & logic
  - route_query routing logic
  - Graph structure (nodes + edges), compiled with StateGraph

WHAT WAS NECESSARILY ADAPTED (and why):
  - `build_retrievers(pdf_path)` read a single hardcoded local file path.
    That can't work in a multi-user upload UI, so it's generalized into
    `build_retrievers_from_docs(docs, ...)` which takes already-loaded
    LangChain Documents. The splitting / embedding / vector-store /
    retriever configuration is 100% identical.
  - Added `load_pdf_docs()` (uploaded file -> PyPDFLoader, same loader
    class as before) and `load_url_docs()` (new: WebBaseLoader, since the
    original had no URL support) so users can supply PDFs and/or URLs.
  - Added a `sources` field to State (additive only) so the two RAG nodes
    can also hand back page/URL metadata for citation display in the UI.
    The retrieval + context-building logic itself is untouched.
  - `llm` / retrievers are now looked up from st.session_state instead of
    module-level globals, since Streamlit reruns the script on every
    interaction and these must persist across reruns without rebuilding.
  - NEW: `widget_gen` counter + "Clear everything" control. Streamlit
    widgets (file_uploader / text_area) can't be cleared by re-assigning
    their value once created — the supported pattern is to change their
    `key`. Bumping `widget_gen` and suffixing every source-widget key with
    it forces fresh, empty widgets on the next rerun, while also resetting
    the indexed retrievers/metadata and the chat transcript.
"""

import os
import tempfile
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ============================================================================
# PAGE CONFIG + THEME (visual only — no backend logic lives here)
# ============================================================================
st.set_page_config(page_title="CourseMate-AI", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

    .stApp { background-color: #0b0f19; color: #e6e6e6; }

    section[data-testid="stSidebar"] {
        background-color: #0e1219;
        border-right: 1px solid #262c3a;
    }

    .cm-tagline {
        color: #7d8590; letter-spacing: 2px; font-size: 0.72rem;
        text-transform: uppercase;
    }
    .cm-title {
        font-family: Georgia, serif; font-size: 2.3rem; color: #f2f2f2;
        margin: 0.2rem 0 0.4rem 0;
    }
    .cm-subtitle { color: #9aa2ad; font-size: 0.95rem; }

    .cm-card {
        background-color: #12172180; border: 1px solid #262c3a;
        border-radius: 10px; padding: 0.9rem 1rem; margin-bottom: 0.5rem;
    }
    .cm-stat-number { color: #e8b04b; font-size: 1.8rem; font-weight: 700; }
    .cm-stat-label {
        color: #7d8590; font-size: 0.68rem; letter-spacing: 1px;
        text-transform: uppercase;
    }

    .stButton>button {
        background-color: #121721; border: 1px solid #e8b04b55;
        color: #e6e6e6; border-radius: 8px; text-align: left;
        font-family: 'JetBrains Mono', monospace;
    }
    .stButton>button:hover { border-color: #e8b04b; color: #e8b04b; }

    .cm-badge {
        display:inline-block; padding: 2px 9px; border-radius: 6px;
        font-size: 0.68rem; letter-spacing:1px; text-transform:uppercase;
        margin-bottom: 6px;
    }
    .cm-badge-academic { background:#1e3a5f; color:#8ec5ff; }
    .cm-badge-fee { background:#3a2f1e; color:#e8b04b; }
    .cm-badge-general { background:#1e3a2a; color:#7ee6a8; }
    .cm-badge-error { background:#3a1e1e; color:#ff8e8e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# CACHED EMBEDDING MODEL
# (cache_resource so it loads once per server, not on every rerun/click —
#  this is what keeps the app from doing heavy work on unrelated UI events)
# ============================================================================
@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


embedding = get_embedding_model()


# ============================================================================
# LLM  (created once from GROQ_API_KEY in the environment / .env file —
# same as the original script's unconditional `llm=ChatGroq(...)`, just
# cached so it isn't re-created on every Streamlit rerun)
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)


try:
    llm = get_llm()
except Exception as e:
    llm = None
    st.error(
        "Could not initialize ChatGroq — make sure GROQ_API_KEY is set in your "
        f".env file, then restart the app. ({e})"
    )

# ============================================================================
# SESSION STATE DEFAULTS
# ============================================================================
DEFAULTS = {
    "chat_history": [],          # UI transcript only (separate from graph invocation)
    "academic_retriever": None,
    "academic_meta": None,
    "fee_retriever": None,
    "fee_meta": None,
    "graph_app": None,
    "programme": "not specified",
    "widget_gen": 0,             # NEW: bump to force-reset file_uploader/text_area widgets
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================================
# STEP 1 — BUILDING RETRIEVERS  (splitter / vector store / retriever kwargs
# are IDENTICAL to the original build_retrievers(); only the document
# source has changed from "hardcoded file path" to "already-loaded docs")
# ============================================================================
def build_retrievers_from_docs(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.25},
    )
    return retriever, len(chunks)


def load_pdf_docs(uploaded_file):
    """Same PyPDFLoader as the original script, applied to an uploaded file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = uploaded_file.name  # friendlier than a temp path
        return docs
    finally:
        os.unlink(tmp_path)


def load_url_docs(url: str):
    """New: URL ingestion (the original script only supported local PDFs)."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = url
    return docs


def index_category(display_name, uploaded_files, url_text, chunk_size, chunk_overlap):
    docs, items = [], []
    pdf_count = url_count = 0

    for f in uploaded_files or []:
        file_docs = load_pdf_docs(f)
        docs.extend(file_docs)
        pdf_count += 1
        items.append({"type": "PDF", "name": f.name, "pages": len(file_docs)})

    for u in [line.strip() for line in (url_text or "").splitlines() if line.strip()]:
        try:
            url_docs = load_url_docs(u)
            docs.extend(url_docs)
            url_count += 1
            items.append({"type": "URL", "name": u, "pages": len(url_docs)})
        except Exception as e:
            st.warning(f"Could not load {u}: {e}")

    if not docs:
        st.warning(f"No sources provided for {display_name}.")
        return None

    retriever, chunk_count = build_retrievers_from_docs(docs, chunk_size, chunk_overlap)
    meta = {"pdf_count": pdf_count, "url_count": url_count, "chunk_count": chunk_count, "items": items}
    return retriever, meta


# ============================================================================
# STEP 2 — STATE  (unchanged, + additive `sources` field for citations)
# ============================================================================
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    classify: str
    retrieved_context: str
    sources: list  # NEW (additive): [{"source": ..., "page": ...}, ...] for UI citations


# ============================================================================
# STEP 3 — NODES  (prompts & logic unchanged; llm/retrievers now resolved
# from st.session_state so they persist correctly across Streamlit reruns)
# ============================================================================
def classifier_node(state: State) -> dict:
    """Look at the following query and classify it to the context which path to take and which mode to select"""
    last_message = state["messages"][-1].content
    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "IMPORTANT: Classify based on the TYPE OF INFORMATION the student is asking for, "
        "not simply because the topic is related to education.\n\n"
        "Use 'academic' ONLY for questions about the college's academic rules, "
        "policies, requirements, or programme-specific information, such as "
        "attendance requirements, exams, grading, credits, promotion, course structure, "
        "summer training requirements, degree requirements, semester rules, or "
        "college academic policies.\n\n"
        "Use 'fee' for questions about tuition, payments, refunds, late charges, "
        "scholarships, fines, fee deadlines, or any college-related money matter.\n\n"
        "Use 'general' for greetings, casual conversation, and general knowledge "
        "questions that can be answered without the college documents. This includes "
        "general educational or technical concepts such as DBMS, Operating Systems, "
        "Computer Networks, OOP, programming, AI, machine learning, mathematics, etc.\n\n"
        "IMPORTANT EXAMPLES:\n"
        "What is DBMS? -> general\n"
        "What is Computer Networks? -> general\n"
        "What is OOP? -> general\n"
        "What is an Operating System? -> general\n"
        "How many semesters are there in BTech? -> academic\n"
        "What is the attendance requirement? -> academic\n"
        "How many credits do I need to graduate? -> academic\n"
        "How much is the tuition fee? -> fee\n"
        "What is the late payment fee? -> fee\n\n"
        f"Query: {last_message}\n\n"
        "Return ONLY one word: academic, fee, or general."
    )
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    return {"classify": category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    retriever = st.session_state.get("academic_retriever")
    if retriever is None:
        return {"retrieved_context": "No academic documents have been indexed yet.", "sources": []}
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [
        {"source": doc.metadata.get("source", "academic handbook"), "page": doc.metadata.get("page")}
        for doc in docs
    ]
    return {"retrieved_context": context, "sources": sources}


def fees_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fees handbook."""
    query = state["messages"][-1].content
    retriever = st.session_state.get("fee_retriever")
    if retriever is None:
        return {"retrieved_context": "No fee documents have been indexed yet.", "sources": []}
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [
        {"source": doc.metadata.get("source", "fee structure"), "page": doc.metadata.get("page")}
        for doc in docs
    ]
    return {"retrieved_context": context, "sources": sources}


def genral_node(state: State) -> dict:
    "Answers directly using the llm's own knowldge ,no retrieval needed"
    return {"retrieved_context": "No retrieval required", "sources": []}


def response_node(state: State) -> dict:
    "Generates the personalized answer , personalized using the student's programme"
    query = state["messages"][-1].content
    programme = state.get("programme", "Unkown")
    context = state["retrieved_context"]
    if context == "No retrieval required":
        prompt = f"""
You are a friendly college assistant.

The student is enrolled in: {programme}.

Answer the user's question accurately using your general knowledge.

Important rules:
- Answer the actual question asked by the student.
- Do NOT assume the student belongs to BCA or any other programme.
- Never mention another programme unless the user explicitly asks about it.
- Mention "{programme}" only when it is genuinely relevant.
- For general educational concepts like DBMS, Computer Networks, OS, OOP, AI, etc., explain them neutrally.
- Do not add irrelevant information about the student's programme.
- Do not ask the student to provide another question if the current question is clear."""
    else:
        prompt = f"""
            You are a college assistant helping a {programme} student.

Use the following official college document context to answer the question.

Rules:
- The student's programme is exactly {programme}.
- Do not assume information belonging to another programme applies to this student.
- If the required information is not present in the context, say that it is not
  available in the provided college documents.
- Do not invent college-specific information .

Context:
{context}

Question:
{query}

Give a clear and precise answer.
     """
    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


def route_query(state: State) -> str:
    if state["classify"] == "academic":
        return "academic_response"
    elif state["classify"] == "fee":
        return "fee_response"
    else:
        return "general"


# ============================================================================
# STEP 5 — GRAPH  (identical structure to the original script)
# ============================================================================
def build_graph():
    graph = StateGraph(State)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_response", academic_rag_node)
    graph.add_node("fee_response", fees_rag_node)
    graph.add_node("general", genral_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_response", "response")
    graph.add_edge("fee_response", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)
    return graph.compile()


# ============================================================================
# QUERY EXECUTION  (only runs on an actual new submission — never on
# unrelated widget interactions like moving a slider or expanding a panel)
# ============================================================================
def run_query(query_text: str):
    if llm is None:
        st.error("GROQ_API_KEY is not set — add it to your .env file and restart the app.")
        return
    if st.session_state.graph_app is None:
        st.session_state.graph_app = build_graph()

    st.session_state.chat_history.append({"role": "human", "content": query_text})

    with st.spinner("Thinking..."):
        try:
            result = st.session_state.graph_app.invoke(
                {"programme": st.session_state.programme, "messages": [("human", query_text)]}
            )
            reply = result["messages"][-1].content
            classify = result.get("classify", "general")
            sources = result.get("sources", [])
        except Exception as e:
            reply = f"Something went wrong while generating the answer: {e}"
            classify = "error"
            sources = []

    st.session_state.chat_history.append(
        {"role": "ai", "content": reply, "classify": classify, "sources": sources}
    )


# ============================================================================
# CLEAR EVERYTHING  (NEW)
# Resets the chat transcript AND both indexed document sets (retrievers +
# metadata), and forces the file_uploader / URL text_area widgets back to
# empty. Streamlit widgets can't be reset by writing to their session_state
# value once instantiated (that raises an exception) — the supported
# pattern is to change the widget's `key`, which is what `widget_gen` does.
# ============================================================================
def clear_everything():
    st.session_state.chat_history = []
    st.session_state.academic_retriever = None
    st.session_state.academic_meta = None
    st.session_state.fee_retriever = None
    st.session_state.fee_meta = None
    st.session_state.graph_app = None
    st.session_state.widget_gen += 1  # changes uploader/text_area keys -> fresh empty widgets


def clear_category_index(state_key: str):
    """Clear just one category's indexed documents (Academic or Fee)."""
    st.session_state[f"{state_key}_retriever"] = None
    st.session_state[f"{state_key}_meta"] = None
    st.session_state.graph_app = None
    st.session_state.widget_gen += 1


# ============================================================================
# SIDEBAR — sources, connection, programme, chunking
# ============================================================================
def render_source_block(display_name, state_key, chunk_size, chunk_overlap):
    st.markdown(f"#### {display_name}")
    gen = st.session_state.widget_gen  # suffix keeps widgets fresh after a clear
    tab_pdf, tab_url = st.tabs(["📄 PDF", "🌐 URL"])
    with tab_pdf:
        files = st.file_uploader(
            f"Upload {display_name} PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"{state_key}_pdf_uploader_{gen}",
        )
    with tab_url:
        url_text = st.text_area(
            f"{display_name} page URLs (one per line)",
            key=f"{state_key}_url_text_{gen}",
            placeholder="https://college.edu/handbook/attendance",
            height=90,
        )

    col_idx, col_clr = st.columns([3, 1])
    if col_idx.button(f"📚 Index {display_name}", key=f"{state_key}_index_btn", use_container_width=True):
        with st.spinner(f"Indexing {display_name} sources..."):
            result = index_category(display_name, files, url_text, chunk_size, chunk_overlap)
        if result:
            retriever, meta = result
            st.session_state[f"{state_key}_retriever"] = retriever
            st.session_state[f"{state_key}_meta"] = meta
            st.success(f"Indexed {meta['chunk_count']} passage(s) for {display_name}.")

    meta = st.session_state.get(f"{state_key}_meta")
    if col_clr.button("🧹", key=f"{state_key}_clear_btn", use_container_width=True, help=f"Clear indexed {display_name} documents"):
        clear_category_index(state_key)
        st.rerun()

    if meta and meta["items"]:
        st.caption("Catalog")
        for item in meta["items"]:
            st.markdown(
                f"<div class='cm-card'>"
                f"<span class='cm-badge cm-badge-{state_key}'>{item['type']}</span> "
                f"<b>{item['name']}</b><br>"
                f"<span style='color:#7d8590;font-size:0.8rem;'>{item['pages']} passage(s) loaded</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


with st.sidebar:
    st.markdown("## 🎓 CourseMate-AI")
    st.caption("Reading room for your academic & fee documents.")

    st.markdown("### 🎓 Your programme")
    st.session_state.programme = st.text_input(
        "e.g. BCA, BTech CSE, MBA", value=st.session_state.programme
    )

    st.markdown("### ⚙️ Chunking settings")
    col_a, col_b = st.columns(2)
    chunk_size = col_a.number_input("Chunk size", min_value=100, max_value=4000, value=800, step=50)
    chunk_overlap = col_b.number_input("Chunk overlap", min_value=0, max_value=1000, value=140, step=10)

    st.divider()
    render_source_block("Academic Handbook", "academic", chunk_size, chunk_overlap)
    st.divider()
    render_source_block("Fee Structure", "fee", chunk_size, chunk_overlap)

    st.divider()
    if st.button("🗑️ Clear everything (chat + indexed documents)", use_container_width=True):
        clear_everything()
        st.rerun()


# ============================================================================
# MAIN AREA — header, stats, suggested questions, transcript, chat input
# ============================================================================
st.markdown(
    "<div class='cm-tagline'>COLLEGE ASSISTANT · GROUNDED IN YOUR ACADEMIC &amp; FEE DOCUMENTS</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='cm-title'>CourseMate-AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='cm-subtitle'>Upload your academic handbook and fee structure "
    "(PDFs and/or URLs), then ask anything — routed to the right document set, with sources cited.</div>",
    unsafe_allow_html=True,
)
st.write("")

academic_meta = st.session_state.academic_meta or {"pdf_count": 0, "url_count": 0, "chunk_count": 0}
fee_meta = st.session_state.fee_meta or {"pdf_count": 0, "url_count": 0, "chunk_count": 0}
total_pdfs = academic_meta["pdf_count"] + fee_meta["pdf_count"]
total_urls = academic_meta["url_count"] + fee_meta["url_count"]
total_passages = academic_meta["chunk_count"] + fee_meta["chunk_count"]

c1, c2, c3 = st.columns(3)
for col, num, label in zip([c1, c2, c3], [total_pdfs, total_urls, total_passages], ["PDFs", "URLs", "Passages indexed"]):
    col.markdown(
        f"<div class='cm-card' style='text-align:center;'>"
        f"<div class='cm-stat-number'>{num}</div><div class='cm-stat-label'>{label}</div></div>",
        unsafe_allow_html=True,
    )

st.write("")

SUGGESTED_QUESTIONS = [
    "What is the attendance requirement?",
    "How many credits do I need to graduate?",
    "What is the tuition fee for this semester?",
    "What is the late payment fee policy?",
    "How many semesters are there in my programme?",
    "Is there any scholarship available?",
    "What is DBMS?",
    "What are the exam and grading rules?",
    "What is the refund policy if I withdraw?",
]

selected_query = None
with st.expander("💡 Suggested questions", expanded=len(st.session_state.chat_history) == 0):
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 3].button(q, key=f"sugg_{i}", use_container_width=True):
            selected_query = q

chat_query = st.chat_input("Ask something about your documents...")
query_to_run = chat_query or selected_query
if query_to_run:
    run_query(query_to_run)

st.write("")

for turn in st.session_state.chat_history:
    if turn["role"] == "human":
        with st.chat_message("user"):
            st.write(turn["content"])
    else:
        with st.chat_message("assistant"):
            classify = turn.get("classify", "general")
            badge_class = {
                "academic": "cm-badge-academic",
                "fee": "cm-badge-fee",
                "general": "cm-badge-general",
                "error": "cm-badge-error",
            }.get(classify, "cm-badge-general")
            st.markdown(
                f"<span class='cm-badge {badge_class}'>{classify.upper()}</span>",
                unsafe_allow_html=True,
            )
            st.write(turn["content"])
            sources = turn.get("sources") or []
            if sources:
                with st.expander("📄 Sources"):
                    for s in sources:
                        page = s.get("page")
                        page_str = f" — page {int(page) + 1}" if isinstance(page, (int, float)) else ""
                        st.markdown(f"- **{s.get('source', 'unknown')}**{page_str}")