from llama_index.readers.confluence import ConfluenceReader
import os, chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from services.llamaindex.embedding import get_embedding_function
from services.langchain.text_to_query import text_to_query
from llama_index.llms.ollama import Ollama



def get_confluence_data_as_vector_llamaindex(url, llm_model, username, password, space_key):
    docs = []

    Settings.llm = Ollama(model=llm_model, request_timeout=1500.0)

    os.environ["CONFLUENCE_USERNAME"] = username
    os.environ["CONFLUENCE_PASSWORD"] = password
    base_url = url
    space_key = space_key

    reader = ConfluenceReader(base_url=base_url)
    documents = reader.load_data(
        space_key=space_key, include_attachments=True, page_status="current"
    )
    for doc in documents:
        # new_doc = Document(text=doc.text, metadata=doc.metadata)
        docs.append(doc.text)
    print(documents)

    index = VectorStoreIndex.from_documents(
        documents, embed_model=get_embedding_function()
    )
    return index, docs

PROMPT_TEMPLATE = """ 
From given above context can you tell me how much percentage the response is matching with the query. Here is my Query and Response, \n
Query: {query}\nResponse: {response}\n
Provide your response like, Matching: [only percentage]
"""

def query_on_confluence_data_llamaindex(llm_model, index, documents, query_text):

    query_engine = index.as_query_engine()
    answer = query_engine.query(query_text)

    
    prompt = PROMPT_TEMPLATE.format(query=query_text, response=answer)

    match llm_model:
        case "mistral":
            LLM_MODEL_NAME = "llama2"
        case "llama2":
            LLM_MODEL_NAME = "mistral"
    matchingIndex = text_to_query(LLM_MODEL_NAME, documents, prompt)
    return {"response": answer.response, "hallucinatingPercentage": matchingIndex.response}


