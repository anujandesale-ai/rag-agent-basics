RAG - Manual Ingestion and Query

## load a web page via WebBaseLoader
1. parse content with bs4.SoupStrainer
2. split text into chunks with RecursiveCharacterTextSplitter
3. index chunks in InMemoryVectorStore
4. query

## perform similarity_search(query, k=4) against the in-memory store
1. build a prompt from the retrieved chunks
2. call the LLM directly with ChatOpenAI.generate(...)
3. print the manual answer

## Commands to Run
create .env and add OPENAI_API_KEY
<br>
cd /rag-agent-basics
<br>
python -m venv .venv-rag-basics
<br>
source .venv-rag-basics/bin/activate
<br>
pip install -r requirements.txt
<br>
python3 rag_ingestion_manual.py

# Accept the query from the command line at runtime

cd <your_workspace>/rag-agent-basics
source .venv/bin/activate
python rag_ingestion_manual.py --query "What is the main idea of this post?"

python rag_ingestion_manual.py --query "What is difference between session and cookie?" --url "https://medium.com/stackademic/token-session-cookie-jwt-oauth2-i-cant-tell-the-difference-529ee28f9055"