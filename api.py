import os
import aiofiles
from fastapi import FastAPI, UploadFile, File
from load import load_database
from load_confluence_langchain import query_on_confluence_data, get_confluence_data_as_vector
from retriever import run_query

app = FastAPI()
DATA_PATH = "data"


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    load_database(DATA_PATH)


@app.post("/ai")
async def get(body: str):
    return run_query(body)

    """
    A function that loads data from Confluence based on a specific query.
    Parameters:
    - body: a string containing the query to be executed
    Returns:
    - the result of the analysis of the query
    """
@app.post("/query-confluence/{space_key}")
async def query_data_from_confluence(body: str):
    q = get_confluence_data_as_vector()
    return query_on_confluence_data(q, body)
