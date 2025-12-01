import os
import logging
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

if __name__ == "__main__":
    logging.info("Faiss implementation script started.")
    loader = PyPDFLoader("9.react_paper.pdf")
    logging.info("Loading documents from PDF...")
    documents = loader.load()
    logging.info(f"Documents loaded successfully. Document count: {len(documents)}")
    
    # Create recursive splitter (not splitting yet)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    logging.info("RecursiveCharacterTextSplitter created (ready to use when needed)")
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if not os.path.exists("faiss_index"):
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logging.info("Vectorstore created successfully.")
        vectorstore.save_local("faiss_index")
    else:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        logging.info("Vectorstore loaded successfully.")
    

    query =  "Give me the gist of react in 3 sentences"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create retriever from vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logging.info("Retriever created with k=3")
    
    # Retrieve context outside the chain
    logging.info("\n=== Retrieving context outside the chain ===")
    retrieved_docs = retriever.invoke(query)
    logging.info(f"Retrieved {len(retrieved_docs)} documents")
    for i, doc in enumerate(retrieved_docs, 1):
        logging.info(f"\nDocument {i} preview: {doc.page_content[:200]}...")
    
    # Format documents function
    def format_docs(docs):
        """Format retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create a Runnable that returns the pre-retrieved documents
    # This allows us to use the already-retrieved docs directly in the chain
    def get_retrieved_docs(_):
        """Return the pre-retrieved documents (ignores input)."""
        return retrieved_docs
    
    rag_prompt = PromptTemplate.from_template(
        """You are a helpful assistant that can answer questions about the document.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.

            Context: {context}

            Question: {input}

            Answer:"""
    )

    # Using pre-retrieved docs directly instead of retriever
    rag_chain = (
        {"context": RunnableLambda(get_retrieved_docs) | format_docs, "input": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    logging.info("\n=== Running RAG chain with pre-retrieved documents ===")
    result = rag_chain.invoke(query)
    logging.info(f"Final Result: {result}")