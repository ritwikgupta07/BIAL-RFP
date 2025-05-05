# ─── IMPORTS ───────────────────────────────────────────────────────────────────
import re
import streamlit as st
from pypdf import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from dotenv import load_dotenv  # Import to load environment variables

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

# ─── AZURE CONFIG ─────────────────────────────────────────────────────────────
load_dotenv()
AZURE_API_TYPE = os.getenv("AZURE_API_TYPE")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")

# ─── INIT EMBEDDINGS & LLM ─────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_API_BASE,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

llm = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_API_BASE,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    temperature=0.0,
    max_tokens=500,
    verbose=False,
)

# ─── STREAMLIT SETUP ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIAL RFP Query Builder", page_icon="bial_logo.png", layout="wide"
)

col1, col2 = st.columns([1, 8], gap="small")
with col1:
    try:
        logo = Image.open("bial_logo.png")
        st.image(logo, width=80)
    except FileNotFoundError:
        pass
with col2:
    st.title("📄 RFP Query Builder")

for key in ("full_text", "chunks", "faiss_store", "faiss_built", "pdfs_hash"):
    st.session_state.setdefault(key, None)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
mode = st.sidebar.radio("Input mode", ["Small Size RFP", "Large Size RFP"])
pdf_files = st.sidebar.file_uploader(
    "Upload Tender PDFs", type="pdf", accept_multiple_files=True
)
excel_file = st.sidebar.file_uploader(
    "Upload Bidder Queries (CSV/XLSX)", type=["csv", "xlsx"]
)

uploaded_pdf_names = [f.name for f in pdf_files] if pdf_files else []
pdfs_hash = hash("".join(uploaded_pdf_names))

if st.session_state.pdfs_hash != pdfs_hash:
    st.session_state.full_text = None
    st.session_state.chunks = None
    st.session_state.faiss_store = None
    st.session_state.faiss_built = False
    st.session_state.pdfs_hash = pdfs_hash

# FAISS controls
if pdf_files and mode == "Large Size RFP":
    cs = st.sidebar.slider("Chunk size", 200, 2000, 600, help="Chars per chunk")
    co = st.sidebar.slider(
        "Chunk overlap", 50, 1000, 150, help="Overlap between chunks"
    )

    if st.sidebar.button("Build Large RFP index"):
        CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)")
        pages = []
        total_pages = 0

        for pdf in pdf_files:
            reader = PdfReader(pdf)
            for i, pg in enumerate(reader.pages, start=1):
                text = pg.extract_text() or ""
                clause = None
                for line in text.splitlines():
                    m = CLAUSE_RE.match(line.strip())
                    if m:
                        clause = m.group(1)
                        break
                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": total_pages + i,
                            "clause": clause,
                            "source_file": pdf.name,
                        },
                    )
                )
            total_pages += len(reader.pages)

        splitter = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=co)
        chunks = splitter.split_documents(pages)

        st.session_state.chunks = chunks
        st.session_state.faiss_store = FAISS.from_documents(chunks, embeddings)
        st.session_state.full_text = "".join(d.page_content for d in pages)
        st.session_state.faiss_built = True
        st.sidebar.success(
            f"Built large RFP index over {len(chunks)} chunks from {len(pdf_files)} files."
        )

    show_excerpts = st.sidebar.checkbox("🔍 Show RFP excerpts", value=False)
    viz_option = st.sidebar.selectbox(
        "Visualization",
        ["None", "Bar chart of response lengths", "Pie chart of answer coverage"],
    )
else:
    show_excerpts = False
    viz_option = "None"

# ─── MAIN PANEL ────────────────────────────────────────────────────────────────
if not pdf_files:
    st.info("Please upload one or more tender PDFs to begin.")
    st.stop()

if mode == "Small Size RFP" and st.session_state.full_text is None:
    full_texts = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        full_texts.append("".join(page.extract_text() or "" for page in reader.pages))
    st.session_state.full_text = "\n".join(full_texts)
    st.success(f"Loaded full text from {len(pdf_files)} PDFs.")

if mode == "Large Size RFP" and not st.session_state.faiss_built:
    st.info("In sidebar: choose chunk settings and click **Build large RFP index**.")
    st.stop()


# ─── ANSWER HELPERS ────────────────────────────────────────────────────────────
def answer_direct(q: str) -> str:
    prompt = (
        "You are BIAL Tender Authority. Using *only* the provided RFP document, "
        "answer in ≤20 words, formal tone. If not specified, reply ‘Not specified in the RFP.’\n\n"
        f"{st.session_state.full_text}\n\nQ: {q}\nA:"
    )
    return llm([HumanMessage(content=prompt)]).content.strip()


def answer_faiss(q: str) -> str:
    docs = st.session_state.faiss_store.similarity_search(q, k=3)
    if show_excerpts:
        for i, d in enumerate(docs, start=1):
            meta = d.metadata
            st.markdown(
                f"**Excerpt {i} (Clause {meta.get('clause')}, pg {meta.get('page')} from {meta.get('source_file')}):** {d.page_content[:200]}…"
            )
    excerpt_text = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "You are BIAL Tender Authority. Using *only* the provided RFP excerpts, "
        "answer in ≤20 words, formal tone.\n\n"
        f"{excerpt_text}\n\nQ: {q}\nA:"
    )
    ans = llm([HumanMessage(content=prompt)]).content.strip()
    top_meta = docs[0].metadata
    clause = top_meta.get("clause")
    page = top_meta.get("page")
    refs = []
    if clause:
        refs.append(f"Clause {clause}")
    if page:
        refs.append(f"pg {page}")
    return f"{ans} ({', '.join(refs)})" if refs else ans


# ─── QUERY UPLOAD & RESPONSES ─────────────────────────────────────────────────
if excel_file:
    df = (
        pd.read_csv(excel_file)
        if excel_file.name.lower().endswith(".csv")
        else pd.read_excel(excel_file)
    )
    if "Query" not in df.columns:
        st.error("Your file must contain a column named **Query**.")
        st.stop()

    st.subheader("Loaded Queries")
    st.dataframe(df, use_container_width=True)

    if st.button("Generate Responses"):
        with st.spinner("Answering queries…"):
            df["Response"] = (
                df["Query"]
                .fillna("")
                .apply(answer_faiss if mode == "Large Size RFP" else answer_direct)
            )

        st.subheader("Responses")
        st.dataframe(df, use_container_width=True)

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Responses", csv_out, "responses.csv", "text/csv")

        if viz_option == "Bar chart of response lengths":
            lengths = df["Response"].str.len().fillna(0)
            fig, ax = plt.subplots()
            ax.bar(df.index.astype(str), lengths)
            ax.set_xlabel("Query #")
            ax.set_ylabel("Response length")
            st.pyplot(fig)

        elif viz_option == "Pie chart of answer coverage":
            covered = (
                df["Response"]
                .str.strip()
                .astype(bool)
                .value_counts()
                .reindex([True, False], fill_value=0)
            )
            fig, ax = plt.subplots()
            ax.pie(covered, labels=["Answered", "Blank"], autopct="%1.0f%%")
            st.pyplot(fig)
else:
    st.info("Please upload your CSV/XLSX of queries.")
