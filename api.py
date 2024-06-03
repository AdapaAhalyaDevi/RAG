import os
import aiofiles

from fastapi import FastAPI, UploadFile, File, Query

from services.langchain.load import load_database as load_db_langchain
from services.llamaindex.load import load_database as load_db_llamaindex
from services.langchain.retriever import run_query as langchain_run_query
from services.llamaindex.retriever import run_query as llamaindex_run_query
from services.langchain.filetoquery import filetoquery as langchain_filetoquery
from services.llamaindex.filetoquery import filetoquery as llamaindex_filetoquery
from services.langchain.load_confluence_langchain import get_confluence_data_as_vector, query_on_confluence_data

app = FastAPI()
DATA_PATH = "data"


@app.post("/langchain/upload")
async def upload_file(file: UploadFile = File(...), project_id: str = None):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    load_db_langchain(DATA_PATH, file.filename, project_id)


@app.post("/llamaindex/upload")
async def upload_file(file: UploadFile = File(...), project_id: str = None):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    load_db_llamaindex(DATA_PATH, file.filename, project_id)




@app.post("/langchain/ai")
async def get(body: str, project_id: str):
    return langchain_run_query(body, project_id)


@app.post("/llamaindex/ai")
async def get(body: str, project_id: str = Query(min_length=1, max_length=15)):
    return llamaindex_run_query(body, project_id)




@app.post("/langchain/file-to-query")
async def upload_file(file: UploadFile = File(...), body: str = None):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    langchain_filetoquery(file.filename, body)


@app.post("/llamaindex/file-to-query")
async def upload_file(file: UploadFile = File(...), body: str = None):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    llamaindex_filetoquery(file.filename, body)




@app.post("/query-confluence/{space_key}")
async def query_data_from_confluence(body: str, url: str, username: str, api_key: str, space_key: str):
    q = get_confluence_data_as_vector(url, username, api_key, space_key)
    return query_on_confluence_data(q, body)

