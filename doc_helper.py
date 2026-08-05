"""
Reads text out of an uploaded file so you don't have to.

Handles .pdf and .txt. You don't need to understand what's in here,
you just need to use it:

    from doc_helper import read_file
    text = read_file(uploaded_file)

Needs:  pip install pypdf
"""

from pypdf import PdfReader


def read_file(uploaded_file) -> str:
    """Takes a file from st.file_uploader and gives back its text."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    return f"Sorry, I can't read {name}. Try a .pdf or .txt file."
