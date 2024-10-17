import os
import io
import uuid
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor

# LangChain and AstraDB imports
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage, Document
from astrapy import DataAPIClient

# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "contract_docs"

# Initialize AstraDB Client
client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(os.environ["ASTRA_DB_ENDPOINT"])
collection = database.get_collection(ASTRA_DB_COLLECTION)

# Streamlit session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_filename' not in st.session_state:
    st.session_state.selected_filename = "ALL"

# Load models
@st.cache_resource(show_spinner='Loading Embedding Model...')
def load_embedding():
    return OpenAIEmbeddings()

@st.cache_resource(show_spinner='Loading Vector Store...')
def load_vectorstore():
    return AstraDBVectorStore(
        embedding=load_embedding(),
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=os.environ["ASTRA_DB_ENDPOINT"]
    )

@st.cache_resource(show_spinner='Loading OpenAI Model...')
def load_model():
    return OpenAI(openai_api_key=OPENAI_API_KEY)

embedding = load_embedding()
vectorstore = load_vectorstore()
llm = load_model()

# Define prompts
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say so. Be concise and clear.

    Context: {context}
    Question: "{question}"
    Answer:
    """
)

CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    """Classify the following text into one of these categories: Legal, Financial, Contract, Corporate, Operational. Only reply with the category name.

    Text: {classify_data}
    Answer:
    """
)

# Functions for PDF text extraction
def extract_text_from_pdf(file):
    file.seek(0)  # Reset file pointer to the start
    pdf = PdfReader(file)
    text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text

def is_scanned_pdf(file):
    file.seek(0)  # Reset file pointer to the start
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        if text.strip():
            return True
    return False

def extract_text_from_scanned_pdf(file):
    file.seek(0)  # Reset file pointer to the start
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
            text += page_text
        return text
    except fitz.EmptyFileError:
        st.error("The uploaded file appears to be empty or invalid.")
        return ""

# Embedding and storing documents
def embed_and_store_text(text, filename, owner, title, category):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={
            "filename": filename, "owner": owner, "title": title, "category": category
        }) for chunk in texts
    ]
    vectorstore.add_documents(documents)
    st.write(f"Stored {len(documents)} chunks from {filename}.")

# Handle user queries and responses
def handle_chat(question, selected_filename):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        context_documents = retrieve_relevant_documents(question, selected_filename)
        if not context_documents:
            response_placeholder.markdown("No relevant content found.")
            return
        context = "\n\n".join(context_documents)
        chain = LLMChain(llm=llm, prompt=ANSWER_PROMPT)
        answer = chain({"context": context, "question": question})["text"]
        response_placeholder.markdown(answer)
        st.session_state.messages.append(AIMessage(content=answer))

def retrieve_relevant_documents(query, selected_filename):
    query_vector = embedding.embed_query(query)
    cursor = collection.find(
        {"metadata.filename": selected_filename},
        sort={"$vector": query_vector},
        projection={"content": True}
    )
    return [doc['content'] for doc in cursor if 'content' in doc]

# Streamlit interface setup
st.title("Contract Management Assistant")
username = st.text_input("Enter Username:")
if not username:
    st.stop()
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    title = st.text_input("Enter PDF Title:")
    if st.button("Process and Store Document"):
        file_bytes = uploaded_file.read()
        file_stream = io.BytesIO(file_bytes)
        is_scanned = is_scanned_pdf(file_stream)
        file_stream.seek(0)  # Reset stream after is_scanned_pdf check
        text = extract_text_from_scanned_pdf(file_stream) if is_scanned else extract_text_from_pdf(file_stream)

        # Classify document category
        chain_category = LLMChain(llm=llm, prompt=CLASSIFY_PROMPT)
        short_text = " ".join(text.split()[:1500])
        category = chain_category({"classify_data": short_text})["text"]
        
        embed_and_store_text(text, uploaded_file.name, username, title, category)
        st.success("Document processed and stored.")

# Query interface
if st.session_state.messages:
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content)

question = st.text_input("Ask a question about a contract:")
if question:
    handle_chat(question, selected_filename="ALL")