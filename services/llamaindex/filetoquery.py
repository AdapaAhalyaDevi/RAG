from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import json 

Settings.llm = Ollama(model="llama2", request_timeout=1500.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def filetoquery(filename, query):
    documents = SimpleDirectoryReader(input_files=[f"././data/{filename}"]).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query(query)

    return {"response": response}