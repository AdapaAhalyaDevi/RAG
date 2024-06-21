import os
import aiofiles

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from pydantic import BaseModel
from ollama_core.upload_file import upload_file
from ollama_core.retriever import run_query
from ollama_core.filetoquery import file_to_query
from ollama_core.load_confluence import load_confluence
from ollama_core.text_to_query import query_from_text
# from services.llamaindex.confluence_to_vector import conf_to_vector
from services.langchain.text_to_m2m import text_to_m2mquery

app = FastAPI()
DATA_PATH = "././data"

ALLOWED_FILE_TYPES = {"text/plain", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/csv", "application/pdf"}



@app.post("/upload")
async def load_file(agent: str = Form(...), project_id: str = Form(...), file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="File type not allowed")

    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    return upload_file(agent, DATA_PATH, file.filename, project_id)




class query(BaseModel):
    agent: str
    llm_model: str
    query: str 
    project_id: str

@app.post("/ai")
async def get(param: query):
    return run_query(param.agent, param.llm_model, param.query, param.project_id)




@app.post("/file-to-query")
async def query_into_document(agent: str = Form(...), llm_model: str = Form(...), file: UploadFile = File(...), query: str = Form(...)):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    return file_to_query(agent, llm_model, DATA_PATH, file.filename, query)




class QueryConfluence(BaseModel):
    agent: str
    llm_model: str
    query: str
    url: str
    username: str
    api_key: str
    space_key: str

@app.post("/query-confluence")
async def confluence_load(param: QueryConfluence):
    response = load_confluence(param.agent, param.llm_model, param.query, param.url, param.username, param.api_key, param.space_key)
    return response


# @app.get("/confluence-vector")
# async def confluence_to_vector():
#     return conf_to_vector()


class TextToQuery(BaseModel):
    agent: str
    llm_model: str
    text_content: str
    query: str

@app.post("/text-to-query")
async def query(param: TextToQuery):
    response = query_from_text(param.agent, param.llm_model, param.text_content, param.query)
    return response




class TextToM2M(BaseModel):
    text: str
    llm_model: str

@app.post("/text-to-m2m")
async def m2m(param: TextToM2M):
    return text_to_m2mquery(param.text, param.llm_model)
