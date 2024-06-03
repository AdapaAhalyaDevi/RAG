from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from services.llamaindex.embedding import get_embedding_function
from llama_index.llms.ollama import Ollama
import chromadb, os

DB_PATH = "././chroma"


def load_database(data_path, filename, project_id):
    documents = load_documents(data_path)
    add_to_db(documents, data_path, project_id, filename)


def load_documents(data_path):
    document_loader = SimpleDirectoryReader(input_dir=f"././{data_path}").load_data()
    return document_loader


def add_to_db(documents, data_path, project_id, filename):

    db = chromadb.PersistentClient(path=f"{DB_PATH}/{project_id}")

    chroma_collection = db.get_or_create_collection(f"{project_id}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=get_embedding_function()
    )

    directory_path = f"././{data_path}"
    if os.path.exists(directory_path):
        try:
            files = os.listdir(directory_path)
            for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except OSError:
            print("Error occurred while deleting files.")

