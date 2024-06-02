import os
import aiofiles

from fastapi import FastAPI, UploadFile, File

from load import load_database
from retriever import run_query

app = FastAPI()
DATA_PATH = "data"


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), project_id: str = None):
    fullpath = os.path.join(DATA_PATH, file.filename)
    async with aiofiles.open(fullpath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    load_database(DATA_PATH, file.filename, project_id)


@app.post("/ai")
async def get(body: str, project_id: str):
    return run_query(body, project_id)
