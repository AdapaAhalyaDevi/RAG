from llama_index.readers.confluence import ConfluenceReader
import os, chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from services.llamaindex.embedding import get_embedding_function
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


def get_confluence_data_as_vector_llamaindex(url, username, password, space_key):
    os.environ["CONFLUENCE_USERNAME"] = username
    os.environ["CONFLUENCE_PASSWORD"] = password
    base_url = url
    space_key = space_key

    reader = ConfluenceReader(base_url=base_url)
    documents = reader.load_data(
        space_key=space_key, include_attachments=True, page_status="current"
    )

    print("------------------")
    print(documents)
    docs = []
    for document in documents:
        title = document.metadata['title']
        content = document.text
        source = document.metadata['url']
        doc_id = document.metadata['page_id']
        url = base_url + doc_id

        data = {
            'title': title,
            'source': source,
            'doc_id': doc_id,
            'id_': doc_id,
            'url': url,
            }


        docs.append(content)

    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    nodes = node_parser.get_nodes_from_documents(
        docs, show_progress=False
    )
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart2")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, embed_model=get_embedding_function()
    )
    return index



def query_on_confluence_data_llamaindex(index, query_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    print(response)




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