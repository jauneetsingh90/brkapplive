import os
import pdfplumber
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore
from util.config import LOGGER, ASTRA_DB_ID, ASTRA_TOKEN
from ragstack_langchain.colbert import ColbertVectorStore as LangchainColbertVectorStore

load_dotenv()

embedding_model = ColbertEmbeddingModel()
database = CassandraDatabase.from_astra(
    astra_token=ASTRA_TOKEN,
    database_id=ASTRA_DB_ID,
    keyspace='default_keyspace'
)

lc_vector_store = LangchainColbertVectorStore(
    database=database,
    embedding_model=embedding_model,
)

# Function to process a single PDF file
def process_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            texts = []
            metadatas = []
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    texts.append(text)
                    metadatas.append({"source": os.path.basename(file_path), "page": page_num + 1})
        if texts:
            # Generate embeddings for each document in batches
            result = lc_vector_store.add_texts(texts=texts, doc_id=os.path.basename(file_path), metadatas=metadatas)
            return result
    except Exception as e:
        LOGGER.error(f"Error processing {file_path}: {e}")
    return None

# Load PDFs and generate ColBERT embeddings in parallel
def load_pdfs_and_generate_embeddings(directory_path):
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_pdf, os.path.join(directory_path, filename))
                   for filename in os.listdir(directory_path) if filename.endswith(".pdf")]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    return results

def main(directory_path):
    results = load_pdfs_and_generate_embeddings(directory_path)
    print("Embeddings generated and stored:", results)

# Replace 'path/to/your/pdf/directory' with the actual path
if __name__ == "__main__":
    pdf_directory = '/Users/jauneetsingh/Downloads/abg'  # Change this to your actual PDF directory path
    main(pdf_directory)