import streamlit as st
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from doc_helper import read_file
import chromadb

#import zipfile
import streamlit as st


db = chromadb.PersistentClient(path="./chroma_db")
brain = db.get_or_create_collection("zeus")
memory = db.get_or_create_collection("zeus_chat")

def anchor_prompt(notes, recalled, questions):
    return f"""ROLE
You are ProsecutorBot, a chatbot that summarizes the infractions committed by a person through their legal files and case files..

CONTEXT
{notes if notes else "(nothing found)"}

EARLIER
{recalled if recalled else "(nothing)"}

RULES
- Use the context above if it exists.
- Use the uploaded zip file as a guideline for the Us legislation, and associate aspects of the case files to this legislation.
- If the answer is not there, and the questions is asking something specific, skip it and say so.
-After each fact, put the specific index and page number of the legislative article you are referring to. 

QUESTION
{questions}
"""

SYSTEM_PROMPT = "You are ProsecutorBot, an AI that summarizes the legal infractions committed by a person after being presented their case files and the history of their crime. You answer questions in a concise and clear manner. Use specific legal jargon but only when necessary. Always remain objective."

def shorten(text, limit=500):
    return text if len(text) <= limit else text[:limit] + " ... rest removed to keep it short"

def chunk_by_sentence(text, max_size = 700):
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_size:
            current += sentence + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def store_document(file):
    text = read_file(file)
    chunks = chunk_by_sentence(text)

    prefix = file.name.replace(" ", "_")
    brain.add(
        documents=chunks,
        metadatas=[{"source": file.name, "chunk": i} for i in range(len(chunks))],
        ids=[f"{prefix}_chunk{i}" for i in range(len(chunks))],
    )
    return len(text), len(chunks)

def remember_exchange(question, answer):
    #Put this Q and A into long term memory so the AI can remember
    memory.add(
        documents=[f"Question: {question}\n Answer: {shorten(answer)}"],
        ids=[f"turn{memory.count()}"]
    )
st.set_page_config(page_title="ProsecutorBOT", page_icon="⚡", layout="wide")

st.title("Welcome to ProsecutorBOT")
st.subheader("A tool to identify legal infractions")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.header("Settings tab")
    with st.form("settings"):
        SYSTEM_PROMPT+= st.text_input("Save custom instructions to the system prompt:")
        sources = st.multiselect("Mood:", ["My first app", "My second app"])
        roles = st.multiselect("Roles:", ["Judge", "Defendant", "Prosecutor"])
        creativity = st.slider("Creativity:", 0.0, 1.0, 0.5)
        THRESHOLD = st.slider("Threshold for accuracy:", 0.0, 3.0, 1.5)
        remember_documents = st.slider("How many chunks to remember", 0, 15, 5)
        remember = st.slider("Recent turns to keep", 0, 10, 3)
        recall = st.slider("Old exchanges to look up", 0, 10, 3)
        notes_only = st.checkbox("Only answer using notes")
        saved = st.form_submit_button("Save")
    if saved:
        st.write(f"saved sources: {sources} and creativity: {creativity}")
    st.caption(f"In memory: {brain.count()} chunks")
    st.caption(f"Long term memory: {memory.count()} exchanges")
    st.caption(f"On screen: {len(st.session_state.messages)} messages")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Forget memory"):
        db.delete_collection("zeus_chat")
        st.rerun()
    if st.button("Forget all documents"):
        db.delete_collection("zeus")
        st.rerun()

for old in st.session_state.messages:
    with st.chat_message(old["role"]):
        st.markdown(old["content"])

user_input = st.chat_input(
    "Ask something here...",
    accept_file=True,
    file_type=["pdf", "txt"],)

if user_input:
    prompt = user_input.text
    prompt_file = None
    if user_input.files:
        prompt_file = user_input.files[0]
    with st.chat_message("user"):
        if prompt_file:
            clean_len, n_chunks = store_document(prompt_file)
            st.write(f"📎 **{prompt_file.name}**")
            st.caption(
                f"{clean_len} characters "
                f"stored as {n_chunks} chunks"
            )
        if prompt:
            st.write(f"{prompt}")
    st.session_state.messages.append(
        {"role": "user", "content": prompt if prompt else f"attached: {prompt_file.name}"}
    )
    with st.chat_message("assistant"):
        if prompt == "Cat Fact":
            r = requests.get("https://catfact.ninja/fact")
            fact = r.json()["fact"]
            answer = fact
            st.write(f"{fact}")
        elif not prompt:
            answer = "Saved. Now ask me something about it!"
            st.write(answer)
        else:
            #1. Anything that is relevant to the uploaded docs:
            notes = ""
            docs, dists, good, metas, used_sources = [], [], [], [], []
            if brain.count() > 0:
                hits = brain.query(query_texts=[prompt], n_results=remember_documents)
                docs = hits["documents"][0]
                dists = hits["distances"][0]
                metas = hits["metadatas"][0]
                for d, s, m in zip(docs, dists, metas):
                    if s < THRESHOLD:
                        good.append(d)
                        used_sources.append(f"{m['source']} (chunk {m['chunk']})")
                notes = "\n\n".join(f"[Source {i+1}] {d}" for i, d in enumerate(good))

            #2. Anything that is relevant to the OLD conversation
            recalled = ""
            old_docs, old_dists, old_good = [], [], []
            if recall > 0 and memory.count() > remember:
                found = memory.query(query_texts=[prompt], n_results=recall)
                old_docs = found["documents"][0]
                old_dists = found["distances"][0]
                old_good = [d for d, s in zip(old_docs, old_dists) if s < THRESHOLD]
                recalled = "\n\n".join(old_good)

            if notes or recalled:
                full_prompt = anchor_prompt(notes, recalled, prompt)
            else:
                full_prompt = prompt

            with st.expander("What I looked up"):
                #1: Notes
                st.caption("From your documents")
                if docs:
                    for d, s, m in zip(docs, dists, metas):
                        mark = "kept " if s < THRESHOLD else "dropped"
                        st.text(f"{s:.3f} {mark} {m['source']} {d[:70]}")
                else:
                    st.text("nothing found")
                #2: Remember past convos
                st.caption("From earlier in our conversation")
                if old_docs:
                    for d, s in zip(old_docs, old_dists):
                        mark = "kept " if s < THRESHOLD else "dropped"
                        st.text(f"{s:.3f} {mark} {d[:70]}")
                else:
                    st.text("nothing found")
                #3: Recall most recent convos
                st.caption("Recent messages I can still see")
                recent = st.session_state.messages[:-1][-(remember * 2):]
                if recent:
                    for m in recent:
                        st.text(f"{m['role']}: {shorten(m['content'], 80)}")
                else:
                    st.text("nothing")

            load_dotenv()
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key = os.environ.get("AI_TOKEN") or st.secrets["AI_TOKEN"],
            )
            #3. The last few turns, word for word bu trimmed
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            past = st.session_state.messages[:-1]
            if remember > 0:
                for m in past[-(remember * 2):]:
                    messages.append({"role": m["role"], "content": shorten(m["content"])})
            messages.append({"role": "user", "content": full_prompt})

            if brain.count() > 0 and not good and not old_good and notes_only:
                answer = "I don't have anything about that in your notes"
                st.write(answer)
            else:
                r = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        temperature=creativity,
                        messages=messages,
                )
                answer = r.choices[0].message.content
                st.write(answer)
                if used_sources:
                    for i, src in enumerate(used_sources):
                        st.caption(f"Source {i+1}: {src}")
        remember_exchange(prompt, answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})