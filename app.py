import streamlit as st
from retriever import hybrid_search
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from config import settings
from streamlit_pdf_viewer import pdf_viewer



st.set_page_config(page_title="RAG Project", layout="wide")

st.title("Reseach Assistant - using RAG")
st.markdown(
    """
    Welcome to your **RAG Assistant**.  
    Ask a question and get precise, context-aware answers from your PDF research papers.
    """
)

col_pdf, col_chat = st.columns([2, 1])

# PDF selection
with col_pdf:
    st.subheader("Source Document")
    pdf_files = os.listdir("data")
    selected_pdf = st.selectbox("Choose a PDF:", pdf_files)
    pdf_path = os.path.join("data", selected_pdf)
    pdf_viewer(pdf_path, width=600, height=600)

with col_chat:
    st.subheader("Ask a Question")
    query = st.text_input("Your query here:")

    llm = ChatGroq(
        model_name=settings.GROQ_MODEL, 
        temperature=0, 
        groq_api_key=settings.GROQ_API_KEY)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context below."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    if st.button("Get Answer") and query.strip():
        with st.spinner("Retrieving documents..."):
            docs = hybrid_search(query)

        context = "\n\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context, question=query)

        with st.spinner("Generating answer..."):
            response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Source Context"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Source {i}:** {doc.page_content[:300]}…")
                if doc.metadata:
                    st.caption(f"Metadata: {doc.metadata}")
                st.divider()