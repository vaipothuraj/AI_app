import streamlit as st
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from doc_helper import read_file
import chromadb

# ─────────────────────────────────────────────────────────────────────────
# DB SETUP
# st.cache_resource ensures this only runs ONCE per app process, not on
# every rerun (Streamlit reruns the whole script on every click/input).
# Without this, chromadb's internal singleton registry gets hit with
# repeated PersistentClient() calls and corrupts itself -> KeyError crash.
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return chromadb.PersistentClient(path="./chroma_db")

db = get_db()

# Three collections instead of one:
# - legislation: the civil code / statutes. Persists independently.
# - case_files: the crime description being analyzed. Wipeable per-case.
# - zeus_chat: rolling conversation memory.
# These are NOT cached themselves -- get_or_create_collection is cheap and
# needs to re-run after a "Forget" button deletes a collection, so that the
# next rerun recreates it fresh instead of holding a dead reference.
legislation = db.get_or_create_collection("legislation")
case_files = db.get_or_create_collection("case_files")
memory = db.get_or_create_collection("zeus_chat")


# ─────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# Two clearly delimited sections (LEGISLATION vs CASE FACTS) so the model
# never treats an alleged fact as law, or a statute as something that
# "happened."
# ─────────────────────────────────────────────────────────────────────────
def anchor_prompt(legislation_notes, case_notes, recalled, question):
    return f"""ROLE
You are ProsecutorBot. You identify legal infractions by comparing case facts
against the legislation provided below. You never invent statutes or facts
not present in the sections below.

=== LEGISLATION (authoritative source of law) ===
{legislation_notes if legislation_notes else "(no relevant legislation retrieved)"}

=== CASE FACTS (the incident being analyzed) ===
{case_notes if case_notes else "(no relevant case facts retrieved)"}

=== PRIOR CONVERSATION ===
{recalled if recalled else "(nothing)"}

RULES
- Only cite infractions that are directly supported by the LEGISLATION section above.
- Never treat CASE FACTS as a source of law, and never treat LEGISLATION as a fact
  about what happened.
- For every infraction listed, cite the specific article/section from the
  LEGISLATION chunk that supports it (chunk id is shown in brackets).
- If CASE FACTS don't map to any retrieved legislation, say so explicitly rather
  than guessing at a statute.

QUESTION
{question}
"""


BASE_SYSTEM_PROMPT = (
    "You are ProsecutorBot, an AI that summarizes the legal infractions committed "
    "by a person after being presented their case files and the relevant civil "
    "code / legislation. You answer questions in a concise and clear manner. "
    "Use specific legal jargon but only when necessary. Always remain objective."
)


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def shorten(text, limit=500):
    return text if len(text) <= limit else text[:limit] + " ... rest removed to keep it short"


def chunk_by_sentence(text, max_size=700):
    # NOTE: this splits purely on ". " with no awareness of article/section
    # boundaries or PDF page breaks. Page-level citations in the prompt are
    # only as good as what read_file() preserves from the source PDF -- if
    # it flattens pages to plain text with no markers, the model has no
    # real page info to cite and may guess. If accurate page citations
    # matter, read_file needs to inject a marker like "--- page 4 ---"
    # at extraction time, and this chunker should respect it as a boundary.
    #
    # HARD CAP FIX: if a single "sentence" (no period found for a long
    # stretch -- common in legal PDFs with numbered clauses, odd line
    # breaks, or missing punctuation) is itself longer than max_size, the
    # old version let it through as ONE unbounded chunk. That's what
    # caused the Groq 413 "request too large" error -- a single chunk of
    # an entire document blew past the token budget. Now any oversized
    # sentence gets force-split at max_size regardless of punctuation.
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > max_size:
            # force-split an oversized "sentence" into hard-capped pieces
            chunks.append(sentence[:max_size].strip())
            sentence = sentence[max_size:]
        if len(current) + len(sentence) < max_size:
            current += sentence + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def already_indexed(collection, filename):
    """Check by metadata whether this filename's chunks are already stored,
    so re-uploading the same PDF (e.g. the civil code sitting in the
    uploader across reruns) doesn't crash on duplicate Chroma IDs."""
    existing = collection.get(where={"source": filename})
    return len(existing["ids"]) > 0


def store_document(file, collection, doc_type):
    """Chunk + store a file into the given collection, tagged with doc_type
    metadata so retrieval can be filtered/debugged by source category even
    though legislation and case_files already live in separate collections."""
    if already_indexed(collection, file.name):
        return None, None  # already there, skip re-adding

    text = read_file(file)
    chunks = chunk_by_sentence(text)
    prefix = f"{doc_type}_{file.name}".replace(" ", "_")

    collection.add(
        documents=chunks,
        metadatas=[{"source": file.name, "chunk": i, "doc_type": doc_type} for i in range(len(chunks))],
        ids=[f"{prefix}_chunk{i}" for i in range(len(chunks))],
    )
    return len(text), len(chunks)


def cap_tokens_approx(text, max_chars=8000):
    """Hard safety net for the Groq TPM limit (12000 tokens on the free
    tier). ~4 chars/token is a rough rule of thumb for English, so 8000
    chars is a conservative ~2000 token ceiling per notes block. This
    catches oversized input even if chunk_by_sentence lets something
    through (e.g. a single word that's still huge, or too many chunks
    retrieved at once)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated to fit token limit]..."


def remember_exchange(question, answer):
    memory.add(
        documents=[f"Question: {question}\n Answer: {shorten(answer)}"],
        ids=[f"turn{memory.count()}"],
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + SESSION STATE
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ProsecutorBOT", page_icon="⚡", layout="wide")

st.title("Welcome to ProsecutorBOT")
st.subheader("A tool to identify legal infractions")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "custom_instructions" not in st.session_state:
    st.session_state["custom_instructions"] = ""


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 Legislation (Civil Code)")
    leg_files = st.file_uploader(
        "Upload statute / civil code PDFs",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="leg_uploader",
    )
    for f in leg_files or []:
        clean_len, n_chunks = store_document(f, legislation, "legislation")
        if clean_len:
            st.caption(f"✅ {f.name}: {n_chunks} chunks indexed")
        else:
            st.caption(f"↺ {f.name}: already indexed")

    st.header("📁 Case Description")
    case_uploads = st.file_uploader(
        "Upload case files to analyze",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="case_uploader",
    )
    for f in case_uploads or []:
        clean_len, n_chunks = store_document(f, case_files, "case")
        if clean_len:
            st.caption(f"✅ {f.name}: {n_chunks} chunks indexed")
        else:
            st.caption(f"↺ {f.name}: already indexed")

    st.divider()
    st.header("Settings")
    with st.form("settings"):
        custom_instructions_input = st.text_input(
            "Add custom instructions to the system prompt:",
            value=st.session_state["custom_instructions"],
        )
        roles = st.multiselect("Roles:", ["Judge", "Defendant", "Prosecutor"])
        creativity = st.slider("Creativity:", 0.0, 1.0, 0.5)
        THRESHOLD = st.slider("Distance threshold for relevance:", 0.0, 3.0, 1.5)
        remember_documents = st.slider("How many chunks to retrieve per collection", 0, 15, 5)
        remember = st.slider("Recent turns to keep", 0, 10, 3)
        recall = st.slider("Old exchanges to look up", 0, 10, 3)
        notes_only = st.checkbox("Only answer using retrieved notes")
        saved = st.form_submit_button("Save")
    if saved:
        # Persist across reruns via session_state, instead of mutating a
        # module-level string in place (which caused unpredictable growth
        # depending on rerun timing in the original code).
        st.session_state["custom_instructions"] = custom_instructions_input
        st.write(f"Saved. Roles: {roles}, creativity: {creativity}")

    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT
    if st.session_state["custom_instructions"]:
        SYSTEM_PROMPT += "\n\nADDITIONAL INSTRUCTIONS:\n" + st.session_state["custom_instructions"]

    st.divider()
    st.caption(f"Legislation: {legislation.count()} chunks")
    st.caption(f"Case files: {case_files.count()} chunks")
    st.caption(f"Long term memory: {memory.count()} exchanges")
    st.caption(f"On screen: {len(st.session_state.messages)} messages")

    # Two independent "forget" actions -- this is what lets you keep the
    # civil code loaded while clearing out a finished case, or vice versa.
    if st.button("Forget case files only"):
        db.delete_collection("case_files")
        st.rerun()
    if st.button("Forget legislation"):
        db.delete_collection("legislation")
        st.rerun()
    if st.button("Forget chat memory"):
        db.delete_collection("zeus_chat")
        st.rerun()
    if st.button("Clear chat display"):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────────────────────────────────
for old in st.session_state.messages:
    with st.chat_message(old["role"]):
        st.markdown(old["content"])

user_input = st.chat_input("Ask something here...")

if user_input:
    prompt = user_input

    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if prompt == "Cat Fact":
            r = requests.get("https://catfact.ninja/fact")
            fact = r.json()["fact"]
            answer = fact
            st.write(fact)

        else:
            # ── 1. Retrieve from LEGISLATION collection ──
            notes_legislation = ""
            leg_docs, leg_dists, leg_good, leg_used_sources = [], [], [], []
            if legislation.count() > 0:
                hits = legislation.query(query_texts=[prompt], n_results=remember_documents)
                leg_docs = hits["documents"][0]
                leg_dists = hits["distances"][0]
                leg_metas = hits["metadatas"][0]
                for d, s, m in zip(leg_docs, leg_dists, leg_metas):
                    if s < THRESHOLD:
                        leg_good.append(d)
                        leg_used_sources.append(f"{m['source']} (chunk {m['chunk']})")
                notes_legislation = "\n\n".join(
                    f"[{leg_used_sources[i]}] {d}" for i, d in enumerate(leg_good)
                )

            # ── 2. Retrieve from CASE_FILES collection ──
            notes_case = ""
            case_docs, case_dists, case_good, case_used_sources = [], [], [], []
            if case_files.count() > 0:
                hits = case_files.query(query_texts=[prompt], n_results=remember_documents)
                case_docs = hits["documents"][0]
                case_dists = hits["distances"][0]
                case_metas = hits["metadatas"][0]
                for d, s, m in zip(case_docs, case_dists, case_metas):
                    if s < THRESHOLD:
                        case_good.append(d)
                        case_used_sources.append(f"{m['source']} (chunk {m['chunk']})")
                notes_case = "\n\n".join(
                    f"[{case_used_sources[i]}] {d}" for i, d in enumerate(case_good)
                )

            # ── 3. Retrieve from chat MEMORY ──
            # Guarded on recall > 0 so we never call query(n_results=0),
            # which Chroma rejects.
            recalled = ""
            old_docs, old_dists, old_good = [], [], []
            if recall > 0 and memory.count() > 0:
                found = memory.query(query_texts=[prompt], n_results=min(recall, memory.count()))
                old_docs = found["documents"][0]
                old_dists = found["distances"][0]
                old_good = [d for d, s in zip(old_docs, old_dists) if s < THRESHOLD]
                recalled = "\n\n".join(old_good)

            # Cap each block independently before assembling the prompt --
            # this is what actually prevents the 413 "request too large"
            # error, on top of the chunking fix above.
            notes_legislation = cap_tokens_approx(notes_legislation, max_chars=6000)
            notes_case = cap_tokens_approx(notes_case, max_chars=4000)
            recalled = cap_tokens_approx(recalled, max_chars=1500)

            full_prompt = anchor_prompt(notes_legislation, notes_case, recalled, prompt)

            # ── Debug/inspection panel ──
            with st.expander("What I looked up"):
                st.caption("From legislation")
                if leg_docs:
                    for d, s, m in zip(leg_docs, leg_dists, leg_metas):
                        mark = "kept  " if s < THRESHOLD else "dropped"
                        st.text(f"{s:.3f} {mark} {m['source']} :: {d[:70]}")
                else:
                    st.text("nothing found")

                st.caption("From case files")
                if case_docs:
                    for d, s, m in zip(case_docs, case_dists, case_metas):
                        mark = "kept  " if s < THRESHOLD else "dropped"
                        st.text(f"{s:.3f} {mark} {m['source']} :: {d[:70]}")
                else:
                    st.text("nothing found")

                st.caption("From earlier in our conversation")
                if old_docs:
                    for d, s in zip(old_docs, old_dists):
                        mark = "kept  " if s < THRESHOLD else "dropped"
                        st.text(f"{s:.3f} {mark} {d[:70]}")
                else:
                    st.text("nothing found")

                st.caption("Recent messages I can still see")
                recent = st.session_state.messages[:-1][-(remember * 2):] if remember > 0 else []
                if recent:
                    for m in recent:
                        st.text(f"{m['role']}: {shorten(m['content'], 80)}")
                else:
                    st.text("nothing")

            # ── LLM call ──
            load_dotenv()
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("AI_TOKEN") or st.secrets["AI_TOKEN"],
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            past = st.session_state.messages[:-1]
            if remember > 0:
                for m in past[-(remember * 2):]:
                    messages.append({"role": m["role"], "content": shorten(m["content"])})
            messages.append({"role": "user", "content": full_prompt})

            if notes_only and not leg_good and not case_good and not old_good:
                answer = "I don't have anything relevant in the legislation or case files for that."
                st.write(answer)
            else:
                r = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=creativity,
                    messages=messages,
                )
                answer = r.choices[0].message.content
                st.write(answer)

                if leg_used_sources:
                    st.caption("Legislation cited: " + ", ".join(leg_used_sources))
                if case_used_sources:
                    st.caption("Case file(s) referenced: " + ", ".join(case_used_sources))

        remember_exchange(prompt, answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
