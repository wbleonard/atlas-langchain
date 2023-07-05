import argparse
import params
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings

# Process arguments
parser = argparse.ArgumentParser(description='Atlas Vector Search Demo')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()

if args.question is None:
    # Some questions to try...
    query = "How big is AT&T?"
    query = "Who founded AT&T?"
    query = "Where is AT&T headquartered?"
    query = "What venues are AT&T branded?"
else:
    query = args.question

print("\nYour question:")
print("--------------")
print(query)

# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection, OpenAIEmbeddings(openai_api_key=params.openai_api_key), index_name=params.index_name
)

# perform a similarity search between the embedding of the query and the embeddings of the documents
print("\nAIs answer:")
print("-------------")
docs = vectorStore.similarity_search(query)

print(docs[0].page_content)
