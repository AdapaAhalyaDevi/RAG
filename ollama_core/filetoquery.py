from services.langchain.filetoquery import filetoquery as langchain_query
from services.llamaindex.filetoquery import filetoquery as llamaindex_query

def file_to_query(agent, llm_model, data_path, filename, query):
    match agent:
        case "langchain":
            return langchain_query(llm_model, data_path, filename, query)

        case "llamaindex":
            return llamaindex_query(llm_model, filename, query)

        case _:
            return {"response": "agent not found"}
