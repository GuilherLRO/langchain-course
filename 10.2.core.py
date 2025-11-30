from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

index_name = "langchain-doc-helper"

def format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def run_llm_chain(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Context: {context}"""),
        ("human", "{input}"),
    ])
    
    # Build RAG chain using LCEL that returns both retrieved docs and answer
    chain = RunnableParallel(
        {
            "retrieved_docs": retriever,
            "answer": (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        }
    )
    
    return chain.invoke(query)


if __name__ == "__main__":
    query = "What is langchain?"
    result = run_llm_chain(query)

    # Highlight the answer with surrounding spaces and ANSI color (yellow)
    answer = result["answer"]
    highlight = f"\n{' '*5}\033[1;33m{answer}\033[0m{' '*5}\n"
    print("Answer:", highlight)
    
    print(f"\nRetrieved {len(result['retrieved_docs'])} documents:")
    
    for i, doc in enumerate(result["retrieved_docs"], 1):
        print(f"\nDocument {i} (source: {doc.metadata.get('source', 'unknown')}):")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    