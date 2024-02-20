
# !mkdir data
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
import fitz
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

file_path = "../data/dev_report.pdf"
doc = fitz.open(file_path)

doc_text = ""
for doc_idx, page in enumerate(doc):
    page_text = page.get_text("text")
    doc_text += page_text


documents = [Document(text = doc_text)]
parser = SentenceSplitter(chunk_size=1024)
nodes = parser.get_nodes_from_documents(documents)

import os

from llama_index.embeddings.openai import OpenAIEmbedding

api_key = "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"
os.environ['OPENAI_API_KEY'] =api_key
embed_model = OpenAIEmbedding(api_key=api_key)



for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content()
    )
    node.embedding = node_embedding

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex(nodes)


query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("What is creative destruction according to the document?")
response_text = response.response

# And we can see where this came from with 
response.source_nodes[0]

# 
# index.set_index_id("vector_index")
# index.storage_context.persist("./storage")
# storage_context = StorageContext.from_defaults(persist_dir="storage")
# # load index
# index = load_index_from_storage(storage_context, index_id="vector_index")