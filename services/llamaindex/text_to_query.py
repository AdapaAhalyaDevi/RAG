from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def text_to_query(llm_model: str, text: str, query: str):

    Settings.llm = Ollama(model = llm_model)

    prompt = PROMPT_TEMPLATE.format(context=text, question=query)

    response_text = Ollama().complete(prompt)
    
    return {"response": response_text}