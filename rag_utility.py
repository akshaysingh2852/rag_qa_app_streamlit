import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
import pdf2image
import pytesseract

# Load environment variables from .env file
load_dotenv()

# Find relative path of current file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the embedding model
embedding = HuggingFaceEmbeddings()

# Load Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.0
)

def load_pdf_with_ocr(file_path):
    # Try PyPDFLoader first for fast digital text extraction
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    total_text = "".join([doc.page_content for doc in docs]).strip()
    
    # If text extraction yielded enough content, return normal documents
    if len(total_text) > 50:
        return docs
    
    # Fallback to OCR using pdf2image and pytesseract for scanned PDFs
    images = pdf2image.convert_from_path(file_path)
    ocr_docs = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        if text.strip():
            ocr_docs.append(Document(page_content=text, metadata={"page": i}))
            
    return ocr_docs

def process_document_to_chroma_db(file_name):
    file_path = f"{working_dir}/docs_dir/{file_name}"
    
    # Load the PDF document with automatic OCR fallback
    documents = load_pdf_with_ocr(file_path)

    # Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    if not texts:
        raise ValueError(
            f"No text could be extracted from '{file_name}' even with OCR. "
            "Please ensure the PDF contains readable text or clear images."
        )
    
    # Store the document chunks in a Chroma vector database
    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0

def answer_question(user_question):
    # Load the persistent Chroma vector database
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    
    # Create a retriever for document search
    retriever = vectordb.as_retriever()

    # Create a RetrievalQA chain to answer user questions using Gemini
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer