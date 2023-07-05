# https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import BSHTMLLoader
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import params
import utils
import sys

# Step 1: Load
filename = "wiki-att.html"
utils.save_web_page_as_html("https://en.wikipedia.org/wiki/AT%26T", filename)
loader = BSHTMLLoader(filename)
data = loader.load()

# Step 2: Transform
text_splitter = CharacterTextSplitter(separator="\n",
                                      chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# Step 3: Embed
# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# Insert the documents in MongoDB Atlas with their embedding
# https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/mongodb_atlas.py
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.index_name
)
