
import streamlit as st

from src.document_loader import extract_text
from src.text_chunker import chunk_text
from src.vector_store import store_chunks
from src.retriever import retrieve_chunks
from src.reranker import rerank_chunks
from src.generator import generate_answer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Mini RAG App",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# SESSION STATE
# ==========================================

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "stored_count" not in st.session_state:
    st.session_state.stored_count = 0

if "ingestion_complete" not in st.session_state:
    st.session_state.ingestion_complete = False


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("📚 Mini RAG Application")

st.markdown(
    """
    Upload your PDF or TXT document.

    The application will extract text, generate chunks,
    create embeddings and store them in Qdrant.
    """
)

st.divider()


# ==========================================
# DOCUMENT UPLOAD
# ==========================================

st.subheader("1. Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT document",
    type=["pdf", "txt"]
)


# ==========================================
# DOCUMENT INGESTION PIPELINE
# ==========================================

if uploaded_file is not None:

    st.info(f"Selected file: {uploaded_file.name}")

    if st.button(
        "Process and Store Document",
        type="primary"
    ):

        # Reset previous processing results
        st.session_state.extracted_text = ""
        st.session_state.chunks = []
        st.session_state.processed_file = None
        st.session_state.stored_count = 0
        st.session_state.ingestion_complete = False

        try:

            # STEP 1: EXTRACT TEXT

            with st.spinner("Extracting document text..."):

                extracted_text = extract_text(uploaded_file)

                if not extracted_text or not extracted_text.strip():
                    raise ValueError(
                        "No text could be extracted from the document."
                    )

            st.success("Text extracted successfully!")

            # STEP 2: GENERATE CHUNKS

            with st.spinner("Generating text chunks..."):

                chunks = chunk_text(
                    extracted_text,
                    chunk_size=1000,
                    chunk_overlap=200
                )

                if not chunks:
                    raise ValueError(
                        "No chunks were generated."
                    )

            st.success(
                f"Generated {len(chunks)} text chunks!"
            )

            # STEP 3: GENERATE EMBEDDINGS
            # STEP 4: STORE VECTORS IN QDRANT

            with st.spinner(
                f"Generating embeddings and storing "
                f"{len(chunks)} chunks in Qdrant..."
            ):

                stored_count = store_chunks(
                    chunks=chunks,
                    document_name=uploaded_file.name
                )

            if stored_count != len(chunks):
                raise ValueError(
                    "Stored vector count does not match chunk count."
                )

            # SAVE SUCCESSFUL RESULTS

            st.session_state.extracted_text = extracted_text
            st.session_state.chunks = chunks
            st.session_state.processed_file = uploaded_file.name
            st.session_state.stored_count = stored_count
            st.session_state.ingestion_complete = True

            st.success(
                "Document ingestion completed successfully!"
            )

        except Exception as e:

            st.error(f"Processing failed: {str(e)}")


# ==========================================
# DISPLAY RESULTS
# ==========================================

if st.session_state.extracted_text:

    st.divider()

    st.subheader("2. Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Characters",
            f"{len(st.session_state.extracted_text):,}"
        )

    with col2:
        st.metric(
            "Total Chunks",
            len(st.session_state.chunks)
        )

    with col3:
        st.metric(
            "Stored Vectors",
            st.session_state.stored_count
        )

    st.write(
        f"**Document:** {st.session_state.processed_file}"
    )

    # ======================================
    # EXTRACTED TEXT
    # ======================================

    st.divider()

    st.subheader("3. Extracted Text")

    with st.expander("View Extracted Document Text"):

        st.text_area(
            "Extracted Content",
            value=st.session_state.extracted_text,
            height=300,
            disabled=True
        )

    # ======================================
    # GENERATED CHUNKS
    # ======================================

    st.divider()

    st.subheader("4. Generated Text Chunks")

    st.caption(
        "Chunk size: 1000 characters | Overlap: 200 characters"
    )

    for index, chunk in enumerate(st.session_state.chunks):

        with st.expander(f"Chunk {index + 1}"):

            st.write(chunk)

            st.caption(
                f"Characters: {len(chunk)}"
            )

    # ======================================
    # QDRANT STORAGE STATUS
    # ======================================

    st.divider()

    st.subheader("5. Vector Database Storage")

    if st.session_state.ingestion_complete:

        st.success("All chunks successfully stored in Qdrant!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Vectors Stored",
                st.session_state.stored_count
            )

        with col2:
            st.metric(
                "Embedding Dimensions",
                3072
            )

        st.info(
            "Your document is now available in the "
            "Qdrant vector database for retrieval."
        )

# ---------------- CHATBOT SECTION ----------------

st.divider()
st.header("Ask Questions About Your Document")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check whether a document has been processed
if st.session_state.ingestion_complete:

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.write(
                            f"Document: {source['document_name']} | "
                            f"Chunk: {source['chunk_index']}"
                        )

    # Chat input
    user_query = st.chat_input("Ask a question about your document...")

    if user_query:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Retrieve relevant chunks
                    retrieved_chunks = retrieve_chunks(
                        query=user_query,
                        top_k=5
                    )

                    # Rerank retrieved chunks
                    reranked_chunks = rerank_chunks(
                        query=user_query,
                        retrieved_chunks=retrieved_chunks,
                        top_n=3
                    )

                    # Generate final answer
                    answer = generate_answer(
                        query=user_query,
                        retrieved_chunks=reranked_chunks
                    )

                    st.markdown(answer)

                    # Display source information
                    with st.expander("View Sources"):
                        for source in reranked_chunks:
                            st.write(
                                f"Document: {source['document_name']} | "
                                f"Chunk: {source['chunk_index']}"
                            )

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": reranked_chunks
                    })

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)

else:
    st.info("Please upload and process a document before asking questions.")

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📚 Mini RAG App")

    st.markdown("### Development Progress")

    stages = [
        "Environment Setup",
        "Streamlit Setup",
        "Document Upload",
        "Document Extraction",
        "Text Chunking",
        "Gemini Embeddings",
        "Qdrant Collection",
        "Vector Storage",
        "Ingestion Pipeline"
    ]

    for stage in stages:
        st.checkbox(
            stage,
            value=True,
            disabled=True
        )

    st.divider()

    st.markdown("### Configuration")

    st.write("Chunk Size: 1000")
    st.write("Chunk Overlap: 200")
    st.write("Embedding Model: gemini-embedding-001")
    st.write("Dimensions: 3072")
    st.write("Database: Qdrant")
    st.write("Distance: COSINE")

    st.divider()

    st.caption("Mini RAG App | Development Version")
