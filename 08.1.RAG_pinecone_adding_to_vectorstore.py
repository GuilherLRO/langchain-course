import logging
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

if __name__ == "__main__":
    logging.info("RAG Pinecone script started.")
    loader = TextLoader("one_piece_wiki.txt")
    logging.info("Loading documents from one_piece_wiki.txt ...")
    documents = loader.load()
    logging.info(f"Documents loaded successfully. Document count: {len(documents)}")
    logging.debug(f"Loaded documents: {documents}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    logging.info("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Splitting complete. Total chunks: {len(chunks)}")
    
    # Add unique IDs to each chunk's metadata and set document ID
    logging.info("Adding IDs to chunk metadata...")
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk-{i}"
        # Set the document ID (used as Pinecone vector ID to prevent duplicates)
        chunk.id = chunk_id
        # Also add ID to metadata for reference
        if chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata["chunk_id"] = chunk_id
    logging.info(f"IDs added to {len(chunks)} chunks.")
    logging.debug(f"Sample chunk (first): {chunks[0] if chunks else 'No chunks'}")

    # Check how many dimensions each chunk will be represented with
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    sample_text = chunks[0].page_content if chunks else "sample text"
    embedding_vector = embeddings.embed_query(sample_text)
    dimension = len(embedding_vector)
    logging.info(f"Each chunk will be represented by an embedding vector of dimension: {dimension}")

    logging.info("Initialized OpenAI Embeddings and Pinecone VectorStore.")
    vectorstore = PineconeVectorStore(index_name="one-piece-wiki", embedding=embeddings)

    # Add the same chunks twice to ensure duplication (original code)
    vectorstore.add_documents(chunks)