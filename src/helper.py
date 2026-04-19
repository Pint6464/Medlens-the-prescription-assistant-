from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

#extract text from pdfs files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents




def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    given a list of documents objects, return a new list of documents objects
    containing only 'source' in metadata and the original page_content.
    """ 
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


#split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,)
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk



def download_embeddings():
    '''
    download and return the HuggingFace embeddings model.
    '''
    import os
    import sys
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    print("[INFO] Loading HuggingFace embeddings model...", flush=True)
    sys.stdout.flush()
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_folder,
        model_kwargs={"device": "cpu"}
    )
    print("[INFO] Embeddings model loaded successfully!", flush=True)
    sys.stdout.flush()
    return embeddings
