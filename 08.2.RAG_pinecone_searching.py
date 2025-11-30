import os
import logging
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name="one-piece-wiki", embedding=embeddings)

if __name__ == "__main__":
    logging.info("RAG Pinecone searching script started.")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    logging.info(f"Retriever created with k={3}")
    
    # Create prompt template
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""),
        ("human", "{input}"),
    ])
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create document chain
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    # Query
    query = "What is the main character of One Piece?"
    logging.info(f"Query: {query}")
    
    result = retrieval_chain.invoke({"input": query})
    
    logging.info(f"Answer: {result['answer']}")
    
    # Optional: Show retrieved context
    logging.info(f"\nRetrieved {len(result['context'])} documents")
    for i, doc in enumerate(result['context'][:3], 1):
        logging.info(f"\nDocument {i} preview: {doc.page_content[:200]}...")
    counter = 0
    for i, doc in enumerate(result['context'], 1):
        counter += 1
    logging.info(f"Total documents retrieved: {counter}")

        