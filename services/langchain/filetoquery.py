# import
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from services.langchain.embedding import get_embedding_function
import os




PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def filetoquery(llm_model, data_path, filename, query):

    LLM_MODEL_NAME = llm_model

    _, file_extension = os.path.splitext(filename)
    match file_extension:
        case '.txt':
            document_loader = TextLoader(f"{data_path}/{filename}")
        case '.docx' | '.doc':
            document_loader = Docx2txtLoader(f"{data_path}/{filename}")
        case '.pdf':
            document_loader = PyPDFLoader(f"{data_path}/{filename}")
        case '.csv':
            document_loader = CSVLoader(f"{data_path}/{filename}")
        case _:
            print("File Format is Not Supported")
    documents = document_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embedding_function = get_embedding_function()
    db = Chroma.from_documents(docs, embedding_function)
    results = db.similarity_search_with_score(query, k=2)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model = Ollama(model=LLM_MODEL_NAME)
    response_text = model.invoke(prompt)
    
    return {"response": response_text}