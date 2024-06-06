from llama_index.readers.confluence import ConfluenceReader
import os, chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from services.llamaindex.embedding import get_embedding_function


def get_confluence_data_as_vector_llamaindex(url, username, password, space_key):
    os.environ["CONFLUENCE_USERNAME"] = username
    os.environ["CONFLUENCE_PASSWORD"] = password
    base_url = url
    space_key = space_key

    reader = ConfluenceReader(base_url=base_url)
    documents = reader.load_data(
        space_key=space_key, include_attachments=True, page_status="current"
    )

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("default")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=get_embedding_function()
    )
    return index


def query_on_confluence_data_llamaindex(index, query_text):
    query_engine = index.as_query_engine()
    answer = query_engine.query(query_text)
    return {"response": answer.response}

# # create client and a new collection
# chroma_client = chromadb.EphemeralClient()
# chroma_collection = chroma_client.create_collection("quickstart")

# # define embedding function
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# # load documents
# documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# # set up ChromaVectorStore and load in data
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, embed_model=embed_model
# )

# # Query Data
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# display(Markdown(f"{response}"))
