from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import json 

Settings.llm = Ollama(model="llama2", request_timeout=1500.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def filetoquery(filename, query):
    print("-----------------")
    print(filename)
    documents = SimpleDirectoryReader(input_files=[f"././data/{filename}"]).load_data()
    
    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query(query)
    print("get_response", type(response.get_response), response.get_response)
    print("print_response_stream", type(response.print_response_stream), response.print_response_stream)
    print("response_gen", type(response.response_gen), response.response_gen)
    print("response_txt", type(response.response_txt), response.response_txt)
    print(dir(response))
    print(response.print_response_stream())

    return {"response": response}