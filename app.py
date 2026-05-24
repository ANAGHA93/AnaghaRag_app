# app.py

import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

# -----------------------------------
# LOAD ENV VARIABLES
# -----------------------------------

load_dotenv()

# -----------------------------------
# LANGCHAIN IMPORTS
# -----------------------------------

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

# -----------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Dentsu Strict RAG Bot",
    page_icon="📄",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("📄 Dentsu Strict RAG Bot")

st.markdown("""
This assistant answers ONLY from uploaded PDF documents.

If information is unavailable in the uploaded document,
the assistant responds with:

**Information not available in knowledge source.**
""")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF Document",
    type="pdf"
)

# -----------------------------------
# MAIN APPLICATION
# -----------------------------------

if uploaded_file:

    # -----------------------------------
    # SAVE TEMP FILE
    # -----------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        pdf_path = tmp_file.name

    # -----------------------------------
    # LOAD PDF
    # -----------------------------------

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # -----------------------------------
    # SPLIT DOCUMENTS
    # -----------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=400
    )

    docs = text_splitter.split_documents(documents)

    st.success(
        f"PDF Loaded Successfully : {len(docs)} chunks created"
    )

    # -----------------------------------
    # AZURE OPENAI EMBEDDINGS
    # -----------------------------------

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("EMBEDDING_MODEL_NAME"),
        api_version=os.getenv("api_version_embedding")
    )

    # -----------------------------------
    # CREATE VECTOR STORE
    # -----------------------------------

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # -----------------------------------
    # STRICT RETRIEVER
    # -----------------------------------

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # -----------------------------------
    # STRICT PROMPT TEMPLATE
    # -----------------------------------

    strict_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a STRICT enterprise document assistant.

IMPORTANT RULES:

1. Answer ONLY from the provided document context.
2. NEVER use outside knowledge.
3. NEVER guess.
4. NEVER hallucinate.
5. NEVER create information.
6. If answer is NOT present in the context,
   respond EXACTLY with:

Information not available in knowledge source.

7. Keep answers concise and document-based only.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # -----------------------------------
    # AZURE CHAT MODEL
    # -----------------------------------

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("CHAT_MODEL_NAME"),
        api_version=os.getenv("api_version"),
        temperature=0
    )

    # -----------------------------------
    # CREATE STRICT RAG CHAIN
    # -----------------------------------

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": strict_prompt
        }
    )

    # -----------------------------------
    # USER QUERY
    # -----------------------------------

    query = st.chat_input(
        "Ask question from uploaded PDF..."
    )

    # -----------------------------------
    # PROCESS QUERY
    # -----------------------------------

    if query:

        # USER MESSAGE

        with st.chat_message("user"):

            st.markdown(query)

        # ASSISTANT MESSAGE

        with st.chat_message("assistant"):

            with st.spinner("Searching knowledge source..."):

                result = qa_chain({
                    "query": query
                })

                answer = result["result"]

                st.markdown(answer)

                # -----------------------------------
                # SHOW SOURCE DOCUMENTS
                # -----------------------------------

                with st.expander("View Retrieved Sources"):

                    source_docs = result["source_documents"]

                    if source_docs:

                        for i, doc in enumerate(source_docs):

                            st.markdown(f"### Source Chunk {i+1}")

                            st.write(doc.page_content[:1000])

                    else:

                        st.write("No source documents retrieved.")