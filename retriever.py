from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import time, os
from embedding import get_embedding_function

CHROMA_PATH = "database"
# DATA_PATH = "data"
LLM_MODEL_NAME = "llama2"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def run_query(query_text: str, database_name):
    path = f"{CHROMA_PATH}/{database_name}"
    print(database_name)
    if not os.path.exists(path):
        print("Database not found")
        return {"response": f"{database_name} Not Found"}
    
    start_time = time.time()
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=f"{CHROMA_PATH}/{database_name}", embedding_function=embedding_function)
    print("Searching Vector");
    results = db.similarity_search_with_score(query_text, k=5)
    print("Vector Search completed");
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print("Context", context_text);
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("\n\nGenerated Prompt");
    model = Ollama(model=LLM_MODEL_NAME)
    print("\n\nLlama2 Loaded");
    response_text = model.invoke(prompt)
    print("\n\nGot the query response");
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    end_time = time.time()
    print("Time", end_time-start_time)
    return {"response": response_text}
