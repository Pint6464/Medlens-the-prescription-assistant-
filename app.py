from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
from src.prompt import *
import os

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

# Get absolute paths and support both repo layouts:
# 1) templates/static/.env inside the app folder
# 2) templates/static/.env one level above the app folder
app_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(app_dir)

local_template_folder = os.path.join(app_dir, 'templates')
local_static_folder = os.path.join(app_dir, 'static')
local_env_file = os.path.join(app_dir, '.env')

workspace_template_folder = os.path.join(workspace_root, 'templates')
workspace_static_folder = os.path.join(workspace_root, 'static')
workspace_env_file = os.path.join(workspace_root, '.env')

template_folder = local_template_folder if os.path.isdir(local_template_folder) else workspace_template_folder
static_folder = local_static_folder if os.path.isdir(local_static_folder) else workspace_static_folder
env_file = local_env_file if os.path.isfile(local_env_file) else workspace_env_file

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

# Load local .env first, then workspace .env with override so latest valid keys win.
if os.path.isfile(local_env_file):
    load_dotenv(dotenv_path=local_env_file, override=False)
if os.path.isfile(workspace_env_file):
    load_dotenv(dotenv_path=workspace_env_file, override=True)

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

print(f"[DEBUG] PINECONE_API_KEY loaded: {PINECONE_API_KEY[:20] if PINECONE_API_KEY else 'None'}...")
print(f"[DEBUG] GROQ_API_KEY loaded: {GROQ_API_KEY[:20] if GROQ_API_KEY else 'None'}...")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print("[INFO] Starting embeddings download...", flush=True)
embeddings = download_embeddings()
print("[INFO] Embeddings ready!", flush=True)

print("[INFO] Connecting to Pinecone...", flush=True)
index_name = "medlens-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
# Fall back to a local FAISS store if Pinecone is unavailable or the API key is invalid.
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
    print("[SUCCESS] Connected to Pinecone!", flush=True)
except Exception as e:
    print(f"[warning] Pinecone vector store unavailable, falling back to in-memory store: {e}", flush=True)
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda

    fallback_docs = [
        Document(page_content="Hello! Pinecone is not available, so this is a local fallback document.", metadata={"source": "fallback"})
    ]
    retriever = RunnableLambda(lambda _: fallback_docs)
    print("[INFO] Using local fallback retriever", flush=True)

chatModel = ChatGroq(model="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    try:
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])
    except Exception as e:
        details = str(e)
        if "invalid api key" in details.lower() or "401" in details:
            error_msg = f"Error: API key validation failed. Check GROQ_API_KEY and PINECONE_API_KEY. Details: {details}"
        else:
            error_msg = f"Error while generating response: {details}"
        print(f"Chat Error: {error_msg}")
        return error_msg



if __name__ == '__main__':
    print("\n" + "="*50, flush=True)
    print("[SUCCESS] All systems initialized!", flush=True)
    print("[INFO] Starting Flask server on http://0.0.0.0:8080", flush=True)
    print("="*50 + "\n", flush=True)
    app.run(host="0.0.0.0", port=8080, debug=True)
