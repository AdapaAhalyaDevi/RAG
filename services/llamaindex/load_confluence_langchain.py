# Example that reads the pages with the `page_ids`
from llama_index.readers.confluence import ConfluenceReader
import os


token = {"access_token": "", "token_type": "Bearer"}
oauth2_dict = {"client_id": "GrYOOCWksm2mb9MxQRVMyhWdc47fOfhn", "token": token}

base_url = "https://temp-me-rag.atlassian.net/wiki"



# page_ids = ["<page_id_1>", "<page_id_2>", "<page_id_3"]
space_key = "EPR"
os.environ["CONFLUENCE_API_TOKEN"] = ""
# os.environ["CONFLUENCE_USERNAME"] = "tamigek805@mcatag.com"
# os.environ["CONFLUENCE_PASSWORD"] = "$e_6=2vTNxi66F4"
print(api_token1)
reader = ConfluenceReader(base_url=base_url)
documents = reader.load_data(
    space_key=space_key, limit=5,
        max_num_results=4
)
print(documents)
documents.extend(
    reader.load_data(
        page_ids=[], include_children=True, include_attachments=True
    )
)