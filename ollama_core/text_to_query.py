from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama



LLM_MODEL_NAME = "llama2"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def text_to_query(text: str, query: str):

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=text, question=query)

    model = Ollama(model=LLM_MODEL_NAME)
    
    response_text = model.invoke(prompt)

    return {"response": response_text}