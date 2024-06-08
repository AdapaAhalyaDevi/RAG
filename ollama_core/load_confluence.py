from services.langchain.load_confluence import get_confluence_data_as_vector_langchain, query_on_confluence_data_langchain
from services.llamaindex.load_confluence import get_confluence_data_as_vector_llamaindex, query_on_confluence_data_llamaindex

def load_confluence(agent, llm_model, query, url, username, api_key, space_key):
    match agent:
        case "langchain":
            index = get_confluence_data_as_vector_langchain(url, query, username, api_key, space_key)
            response = query_on_confluence_data_langchain(llm_model, index, query)
            return response

        case "llamaindex":
            index, documents = get_confluence_data_as_vector_llamaindex(url, llm_model, query, username, api_key, space_key)
            response = query_on_confluence_data_llamaindex(llm_model, index, documents, query)
            return response

        case _:
            return {"response": "agent not found"}
        
