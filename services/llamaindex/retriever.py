import chromadb, os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from services.llamaindex.embedding import get_embedding_function
from llama_index.llms.ollama import Ollama


CHROMA_PATH = "././chroma"
LLM_MODEL_NAME = "llama2"
Settings.llm = Ollama(model="llama2", request_timeout=1500.0)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def run_query(query_text: str, project_id):
    path = f"{CHROMA_PATH}/{project_id}"
    if not os.path.exists(path):
        print("Database not found")
        return {"response": f"{project_id} Not Found"}

    embedding_function = get_embedding_function()

    db = chromadb.PersistentClient(path=f"{CHROMA_PATH}/{project_id}")
    print(db)
    chroma_collection = db.get_collection(f"{project_id}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embedding_function
    )
    # print(query_text)
    query_engine = index.as_query_engine()
    response_text = query_engine.query(query_text)
    print("-----------------------")
    print(type(response_text))
    print(response_text)

    return {"response": response_text}



