import asyncio
import os
import ssl
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import certifi
from rich.logging import RichHandler
from rich.console import Console
from rich import traceback

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore 
from langchain_tavily import TavilyCrawl,TavilyMap,TavilyExtract

# Configure rich logging with colors
console = Console()
traceback.install()  # Enable rich tracebacks

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=True,
            show_time=True,
            markup=True,
            tracebacks_show_locals=True,
        )
    ]
)

#configuring ssl context to use certifi for ssl verification
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["SSL_CERT_DIR"] = certifi.where()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds=10)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap( max_depth=5, max_breadth=20,max_pages=100)
tavily_crawl = TavilyCrawl()
vetorstore = PineconeVectorStore(index_name="langchain-doc-helper", embedding=embeddings)


async def main():
    logging.info("[bold cyan]🚀 Ingestion script started[/bold cyan]")

    logging.info("[yellow]📡 Starting web crawl...[/yellow]")
    res = tavily_crawl.invoke({"url":"https://docs.langchain.com",
    "max_depth":5,
    "max_breadth":20,
    "max_pages":50, 
    "extra_depth":"advanced",
    "instructions": "content on ai agents or llm chains"
    })

    all_docs = [Document(page_content = result['raw_content'], metadata = {"source": result['url']}) for result in res['results']]
    logging.info(f"[green]✅ Crawled [bold]{len(all_docs)}[/bold] documents[/green]")
    

    logging.info("[yellow]✂️  Splitting documents into chunks...[/yellow]")
    split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    logging.info(f"[green]✅ Split into [bold]{len(split_docs)}[/bold] chunks[/green]")
    
    # Embed documents asynchronously in batches to maintain order and handle retries
    # The embeddings object already has retry_min_seconds=10 configured for automatic retries
    batch_size = 50  # Process in batches to maintain order and handle rate limits
    all_embeddings = []
    
    logging.info(f"[yellow]🔢 Starting async embedding process in batches of [bold]{batch_size}[/bold]...[/yellow]")
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch]
        batch_num = i // batch_size + 1
        total_batches = (len(split_docs) + batch_size - 1) // batch_size
        logging.info(f"[cyan]🔄 Embedding batch [bold]{batch_num}/{total_batches}[/bold] ([bold]{len(batch)}[/bold] chunks)[/cyan]")
        
        # Use async embedding with retries (handled by OpenAIEmbeddings internally)
        # This maintains the original order since we process sequentially
        batch_embeddings = await embeddings.aembed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        logging.info(f"[green]✓ Batch {batch_num} completed[/green]")
    
    logging.info(f"[bold green]✅ Successfully embedded [bold]{len(all_embeddings)}[/bold] chunks (maintaining original order)[/bold green]")
    
    # Add documents to vectorstore using async execution
    # Run add_documents in a thread pool to keep it non-blocking
    # The vectorstore will use the embeddings object which has retry configuration
    logging.info("[yellow]💾 Adding documents to vectorstore...[/yellow]")
    await asyncio.to_thread(vetorstore.add_documents, split_docs)
    logging.info(f"[bold green]✅ Added [bold]{len(split_docs)}[/bold] documents to vectorstore[/bold green]")
    logging.info("[bold cyan]🎉 Ingestion completed successfully![/bold cyan]")

if __name__ == "__main__":
    asyncio.run(main())
