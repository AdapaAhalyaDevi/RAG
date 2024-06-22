from services.langchain.text_to_query import text_to_query as langchain_query


def query_from_text(agent, llm_model, text, query):
    match agent:
        case "langchain":
            return langchain_query(llm_model, text, query)

        case _:
            return {"response" : "agent not found"}
