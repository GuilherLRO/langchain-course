"""
RAG with Pinecone using LangChain Expression Language (LCEL)

This version uses LCEL directly instead of create_retrieval_chain and 
create_stuff_documents_chain helper functions.

PROS of using LCEL directly:
1. More explicit control over the chain flow - you can see exactly what happens at each step
2. More flexible - easier to customize intermediate steps (e.g., filtering, transformation)
3. Better for debugging - can inspect intermediate results at each stage
4. More composable - easier to add/remove steps or create variations
5. Better performance - no overhead from helper function abstractions
6. More modern approach - LCEL is the recommended way in newer LangChain versions
7. Easier to parallelize - can use RunnableParallel for concurrent operations
8. Better streaming support - LCEL chains support streaming out of the box

CONS of using LCEL directly:
1. More verbose - requires writing more code to achieve the same result
2. Steeper learning curve - need to understand LCEL concepts (RunnablePassthrough, pipe operator, etc.)
3. More boilerplate - need to manually format documents and structure the chain
4. Less beginner-friendly - helper functions abstract away complexity
5. More error-prone - manual chain construction can lead to mistakes
6. Less standardized - different developers might structure chains differently
7. Need to handle document formatting manually (e.g., format_docs function)
8. More code to maintain - more moving parts to keep track of

When to use LCEL:
- When you need fine-grained control over the chain
- When building complex, multi-step workflows
- When you want to customize intermediate processing
- When performance is critical
- When you're comfortable with LangChain's advanced features

When to use helper functions:
- When building simple, standard RAG pipelines
- When learning LangChain (easier to get started)
- When you want less code and faster prototyping
- When the standard flow is sufficient for your needs
"""

import os
import logging
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name="one-piece-wiki", embedding=embeddings)

if __name__ == "__main__":
    logging.info("RAG Pinecone searching script started (LCEL version).")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    logging.info(f"Retriever created with k={3}")
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""),
        ("human", "{input}"),
    ])
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Format documents function
    def format_docs(docs):
        """Format retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build RAG chain using LCEL (LangChain Expression Language)
    # The chain: input -> retrieve -> format -> prompt -> llm -> parse
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Query
    query = "What is the main character of One Piece?"
    logging.info(f"Query: {query}")
    
    # Invoke the chain
    answer = rag_chain.invoke(query)
    logging.info(f"Answer: {answer}")
    
    # Optional: Show retrieved context separately
    retrieved_docs = retriever.invoke(query)
    logging.info(f"\nRetrieved {len(retrieved_docs)} documents")
    for i, doc in enumerate(retrieved_docs[:3], 1):
        logging.info(f"\nDocument {i} preview: {doc.page_content[:200]}...")
    logging.info(f"Total documents retrieved: {len(retrieved_docs)}")

