import os
import aiofiles

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from pydantic import BaseModel
from services.langchain.load import load_database as load_db_langchain
from services.llamaindex.load import load_database as load_db_llamaindex
from services.langchain.retriever import run_query as langchain_run_query
from services.llamaindex.retriever import run_query as llamaindex_run_query
from services.langchain.filetoquery import filetoquery as langchain_filetoquery
from services.llamaindex.filetoquery import filetoquery as llamaindex_filetoquery
from services.langchain.load_confluence import get_confluence_data_as_vector_langchain, query_on_confluence_data_langchain
from services.llamaindex.load_confluence import get_confluence_data_as_vector_llamaindex, query_on_confluence_data_llamaindex
from ollama_core.text_to_query import text_to_query

app = FastAPI()
DATA_PATH = "././data"

ALLOWED_FILE_TYPES = {"text/plain", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/csv", "application/pdf"}



@app.post("/langchain/upload")
async def upload_file(project_id: str = Form(...), file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="File type not allowed")

    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    return load_db_langchain(DATA_PATH, file.filename, project_id)


@app.post("/llamaindex/upload")
async def upload_file(project_id: str = Form(...), file: UploadFile = File(...)):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    load_db_llamaindex(DATA_PATH, file.filename, project_id)


class query(BaseModel):
    body: str 
    project_id: str


@app.post("/langchain/ai")
async def get(param: query):
    return langchain_run_query(param.body, param.project_id)


@app.post("/llamaindex/ai")
async def get(param: query):
    return llamaindex_run_query(param.body, param.project_id)




@app.post("/langchain/file-to-query")
async def file_to_query_langchain(file: UploadFile = File(...), query: str = Form(...)):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    return langchain_filetoquery(DATA_PATH, file.filename, query)


@app.post("/llamaindex/file-to-query")
async def file_to_query_llamaindex(file: UploadFile = File(...), query: str = Form(...)):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    return llamaindex_filetoquery(file.filename, query)



class QueryConfluenceLangchain(BaseModel):
    body: str
    url: str
    username: str
    api_key: str
    space_key: str

@app.post("/langchain/query-confluence")
async def query_data_from_confluence(param: QueryConfluenceLangchain):
    q = get_confluence_data_as_vector_langchain(param.url, param.username, param.api_key, param.space_key)
    return query_on_confluence_data_langchain(q, param.body)


class QueryConfluenceLlamaindex(BaseModel):
    body: str
    url: str
    username: str
    password: str
    space_key: str

@app.post("/llamaindex/query-confluence")
async def query_data_from_confluence(param: QueryConfluenceLlamaindex):
    q = get_confluence_data_as_vector_llamaindex(param.url, param.username, param.password, param.space_key)
    return query_on_confluence_data_llamaindex(q, param.body)



class TextToQuery(BaseModel):
    text_content: str
    query: str

@app.post("/text-to-query")
async def query(param: TextToQuery):
    response = text_to_query(param.text_content, param.query)
    return response
