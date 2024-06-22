from services.langchain.retriever import run_query as langchain_query
from services.llamaindex.retriever import run_query as llamaindex_query


def run_query(agent, llm_model, query_text, project_id):
    match agent:
        case "langchain":
            return langchain_query(llm_model, query_text, project_id)

        case "llamaindex":
            return llamaindex_query(llm_model, query_text, project_id)

        case _:
            return {"response": "agent not found"}
