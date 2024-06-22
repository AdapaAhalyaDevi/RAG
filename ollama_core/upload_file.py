from services.langchain.load import load_database as langchain_ld
from services.llamaindex.load import load_database as llama_index_ld

def upload_file(agent, data_path, filename, project_id):
    match agent:
        case "langchain":
            response_text = langchain_ld(data_path, filename, project_id)
            return response_text

        case "llamaindex":
            response_text = llama_index_ld(data_path, filename, project_id)
            return response_text

        case _:
            return {"response": "agent not found"}
                