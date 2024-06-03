from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import os
from services.langchain.embedding import get_embedding_function

CHROMA_PATH = "././chroma"
LLM_MODEL_NAME = "llama2"


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
    db = Chroma(persist_directory=f"{CHROMA_PATH}/{project_id}", embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model=LLM_MODEL_NAME)
    
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return {"response": response_text}
