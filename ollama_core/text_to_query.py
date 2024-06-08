from services.langchain.text_to_query import text_to_query as langchain_query
from services.llamaindex.text_to_query import text_to_query as llamaindex_query


def query_from_text(agent, llm_model, text, query):
    try:
        match agent:
            case "langchain":
                return langchain_query(llm_model, text, query)

            case _:
                return {"response" : "agent not found"}

    except Exception as error:
        return {"error": error}