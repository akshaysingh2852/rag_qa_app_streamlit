import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

# Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Create docs_dir if it doesn't exist yet
docs_folder = os.path.join(working_dir, "docs_dir")
os.makedirs(docs_folder, exist_ok=True)

st.title("Gemini-3.5-flash - Document RAG")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Define save path pointing to docs_dir
    save_path = os.path.join(docs_folder, uploaded_file.name)
    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner("Processing document (with OCR fallback if needed)..."):
            process_document_to_chroma_db(uploaded_file.name)
        st.info("Document Processed Successfully")
    except Exception as e:
        st.error(f"Error processing document: {e}")

# Text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    if user_question.strip():
        try:
            with st.spinner("Generating answer..."):
                answer = answer_question(user_question)
            st.markdown("### Gemini-3.5-flash Response : ")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")
    else:
        st.warning("Please enter a question.")