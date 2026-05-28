import os
import bs4
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import HumanMessage

# Load environment variables from .env file
load_dotenv()


def create_model() -> ChatOpenAI:
    print("Step 1: Creating ChatOpenAI model...")
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    print(f"✓ Model created: {model.model_name}")
    return model


def create_embeddings() -> OpenAIEmbeddings:
    print("\nStep 2: Creating embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print("✓ Embeddings created")
    return embeddings


def load_web_documents(urls):
    print("\nStep 3: Loading documents from the web...")
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} document(s)")
    if documents:
        print(f"  Total characters: {len(documents[0].page_content)}")
    return documents


def create_inmemory_vector_store(documents, embeddings):
    """Create an in-memory vector store and add document chunks to it."""
    print("\nStep 4b: Building InMemoryVectorStore in memory...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunk(s)")
    vector_store = InMemoryVectorStore(embeddings)
    document_ids = vector_store.add_documents(chunks)
    print(f"✓ InMemoryVectorStore created — added {len(document_ids)} documents")
    return vector_store

# Without RetrievalQA, you’d have to manually:
# retrieve similar docs,
# build a prompt,
# call the LLM,
# format the answer.
def run_sample_query_manual(model, vector_store, query: str):
    print(f"\nStep 5b: Running manual retrieve/prompt/call for: {query!r}")
    if not hasattr(vector_store, "similarity_search"):
        raise RuntimeError("Vector store does not support similarity_search()")

    docs = vector_store.similarity_search(query, k=4)
    print(f"✓ Retrieved {len(docs)} document(s)")
    context = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source") if hasattr(doc, "metadata") else None
        header = f"Source {i}: {source}" if source else f"Source {i}:"
        context.append(f"{header}\n{doc.page_content}")

    prompt = (
        "Use the following context to answer the question.\n\n"
        + "\n\n".join(context)
        + f"\n\nQuestion: {query}\n\nAnswer concisely based on the provided context."
    )

    messages = [[HumanMessage(content=prompt)]]
    result = model.generate(messages)
    answer = result.generations[0][0].text

    print("\nManual answer:")
    print(answer.strip())
    return answer


def main():
    model = create_model()
    embeddings = create_embeddings()

    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    documents = load_web_documents(urls)
    if not documents:
        raise SystemExit("No documents loaded. Check the URL and HTML parsing rules.")

    vector_store = create_inmemory_vector_store(documents, embeddings)
    run_sample_query_manual(model, vector_store, "What is the main idea of the agent post?")


if __name__ == "__main__":
    main()
