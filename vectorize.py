# https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import params

# Step 1: Load
# https://python.langchain.com/v0.2/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html
loaders = [
 WebBaseLoader("https://en.wikipedia.org/wiki/AT%26T"),
 WebBaseLoader("https://en.wikipedia.org/wiki/Bank_of_America")
]

docs = []

for loader in loaders:
    for doc in loader.lazy_load():
        docs.append(doc)

print('Loaded ' + str(len(docs)) + ' docs')


# Step 2: Transform (Split)
# https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                              "\n\n", "\n", r"(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(docs)
print('Split into ' + str(len(docs)) + ' docs')

# Step 3: Embed
# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
embeddings = OpenAIEmbeddings(openai_api_key=params.OPENAI_API_KEY)

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient(params.MONGODB_CONN_STRING)
collection = client[params.DB_NAME][params.COLL_NAME]

# Reset w/out deleting the Search Index 
collection.delete_many({})

# Insert the documents in MongoDB Atlas with their embedding
# https://python.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.INDEX_NAME
)

# Step 5: Create Vector Search Index
# https://python.langchain.com/v0.2/api_reference/mongodb/index.html
# THIS ONLY WORKS ON DEDICATED CLUSTERS (M10+)
# docsearch.create_vector_search_index(dimensions=1536, update=True)