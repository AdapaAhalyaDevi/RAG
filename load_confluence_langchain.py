import os
from langchain_community.document_loaders import ConfluenceLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from dotenv import load_dotenv
import os
from langchain.schema.document import Document
from embedding import get_embedding_function
from langchain.vectorstores.chroma import Chroma

from langchain.text_splitter import MarkdownHeaderTextSplitter

load_dotenv()

CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_CLIENT_ID = os.getenv("CONFLUENCE_CLIENT_ID")
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

LLM_MODEL_NAME = "llama2"

PROMPT_TEMPLATE = """
The context provided is data from confluence space.
Answer the question based on the context below. 
If the question cannot be answered using the information provided answer
with "It is not clear from the provided data".

Context: {context}

Question: {query}

Answer the question.

Be informative, gentle, and formal. 
Answer:"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


# loading confluence space in to vectorstore

def compute_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks


def get_confluence_data_as_vector():
    loader = ConfluenceLoader(
        url=CONFLUENCE_BASE_URL,
        username=CONFLUENCE_USERNAME,
        api_key=CONFLUENCE_API_TOKEN,
        cql="space = " + CONFLUENCE_SPACE_KEY,
        space_key=CONFLUENCE_SPACE_KEY,
        limit=5,
        max_pages=4
    )
    documents = loader.load()

    docs = []
    for document in documents:
        print("------document---------")
        print(document)
        title = document.metadata['title']
        content = document.page_content
        source = document.metadata['source']
        doc_id = document.metadata['id']
        url = CONFLUENCE_BASE_URL + doc_id

        data = {'title': title,
                'source': source,
                'doc_id': doc_id,
                'url': url,
                'Header 1': '',
                'Header 2': ''
                }
        print(data)
        md_header_splits = markdown_splitter.split_text(content)
        for i, split in enumerate(md_header_splits):
            data['sub_id'] = i
            data.update(split.metadata)

            data[
                'content'] = f"{data['title']}\n\tsubsection:{data['Header 1']}:\n\tsub_subsection:{data['Header 2']}:\n" + split.page_content
            print("-------split markdown----------")
            print(f"conf{doc_id}_{i} - {data}")
            new_doc = Document(page_content=data['content'], metadata=document.metadata)
            docs.append(new_doc)
    return Chroma.from_documents(docs, get_embedding_function())


def query_on_confluence_data(index, query_text):
    results = index.similarity_search_with_score(query_text, k=1)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query_text)

    model = Ollama(model=LLM_MODEL_NAME)

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return {"response": response_text}
