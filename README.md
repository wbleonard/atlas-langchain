
# Semantic Search Made Easy With LangChain and MongoDB

Enabling semantic search on user-specific data is a multi-step process that includes loading, transforming, embedding and storing data before it can be queried. 

![](https://python.langchain.com/v0.1/assets/images/data_connection-95ff2033a8faa5f3ba41376c0f6dd32a.jpg)

That graphic is from the team over at [LangChain](https://python.langchain.com/docs/modules/data_connection/), whose goal is to provide a set of utilities to greatly simplify this process. 

In this tutorial, we'll walk through each of these steps, using MongoDB Atlas as our Store. Specifically, we'll use the [AT&T](https://en.wikipedia.org/wiki/AT%26T) and [Bank of America](https://en.wikipedia.org/wiki/Bank_of_America) Wikipedia pages as our data source. We'll then use libraries from LangChain to Load, Transform, Embed and Store: 

![](./images/flow-store.png)

Once the source is store is stored in MongoDB, we can retrieve the data that interests us:

![](./images/flow-retrieve.png)


## Prerequisites
* [MongoDB Atlas Subscription](https://cloud.mongodb.com/) (Free Tier is fine)
* Open AI [API key](https://platform.openai.com/account/api-keys)

## Quick Start Steps
1. Get the code:
```zsh
git clone https://github.com/wbleonard/atlas-langchain.git
```
2. Update [params.py](params.py) with your MongoDB connection string and Open AI [API key](https://platform.openai.com/account/api-keys).
3. Create a new Python environment
```zsh
python3 -m venv env
```
4. Activate the new Python environment
```zsh
source env/bin/activate
```

5. Install the requirements
```zsh
pip3 install -r requirements.txt
```
6. Load, Transform, Embed and Store
```zsh
python3 vectorize.py
```

7. Retrieve
```zsh
python3 query.py -q "Who started AT&T?"
```

## The Details
### Load -> Transform -> Embed -> Store 
#### Step 1: Load
There's no lacking for sources of data: Slack, YouTube, Git, Excel, Reddit, Twitter, etc., and [LangChain provides a growing list](https://python.langchain.com/v0.2/api_reference/community/document_loaders.html) of integrations that includes this list and many more.

For this exercise, we're going to use the [WebBaseLoader](https://python.langchain.com/v0.2/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html) to load the Wikipedia pages for [AT&T](https://en.wikipedia.org/wiki/AT%26T) and [Bank of America](https://en.wikipedia.org/wiki/Bank_of_America). 

```python
from langchain_community.document_loaders import WebBaseLoader

# Step 1: Load
loaders = [
 WebBaseLoader("https://en.wikipedia.org/wiki/AT%26T"),
 WebBaseLoader("https://en.wikipedia.org/wiki/Bank_of_America")
]

docs = []

for loader in loaders:
    for doc in loader.lazy_load():
        docs.append(doc)

```

 #### Step 2: Transform (Split)
 Now that we have a bunch of text loaded, it needs to be split into smaller chunks so we can tease out the relevant portion based on our search query. For this example we'll use the recommended [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter). As I have it configured, it attempts to split on paragraphs (`"\n\n"`), then sentences(`"(?<=\. )"`), then words (`" "`) using a chunk size of 1000 characters. So if a paragraph doesn't fit into 1000 characters, it will truncate at the next word it can fit to keep the chunk size under 1000 chacters. You can tune the `chunk_size` to your liking. Smaller numbers will lead to more documents, and vice-versa.

```python
# Step 2: Transform (Split)
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                              "\n\n", "\n", r"(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(docs)
```

#### Step 3: Embed
[Embedding](https://python.langchain.com/docs/modules/data_connection/text_embedding/) is where you use an LLM to create a vector representation text. There are many options to choose from, such as [OpenAI](https://openai.com/) and [Hugging Face](https://huggingface.co/), and LangChang provides a standard interface for interacting with all of them. 

For this exercise we're going to use the popular [OpenAI embedding](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html#langchain_openai.embeddings.base.OpenAIEmbeddings). Before proceeding, you'll need an [API key](https://platform.openai.com/account/api-keys) for the OpenAI platform, which you will set in [params.py](params.py).

We're simply going to load the embedder in this step. The real power comes when we store the embeddings in Step 4. 

```python
# Step 3: Embed
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=params.OPENAI_API_KEY)
```

#### Step 4: Store
You'll need a vector database to store the embeddings, and lucky for you MongoDB fits that bill. Even luckier for you, the folks at LangChain have a [MongoDB Atlas module](https://python.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/) that will do all the heavy lifting for you! Don't forget to add your MongoDB Atlas connection string to [params.py](params.py).

```python
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

client = MongoClient(params.MONGODB_CONN_STRING)
collection = client[params.DB_NAME][params.COLL_NAME]

# Insert the documents in MongoDB Atlas with their embedding
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=index_name
)
```

Lastly, we need to created the vector search index, which Langchain can do for us as well:
```python
# Step 5: Create Vector Search Index
docsearch.create_vector_search_index(dimensions=1536)
```

You'll find the complete script in [vectorize.py](vectorize.py), which needs to be run only once or when new data sources are added.

```zsh
python3 vectorize.py
```

### Retrieve 
We could now run a search, using methods like [similirity_search](https://python.langchain.com/v0.2/api_reference/mongodb/vectorstores/langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch.html#langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch.similarity_search) or [max_marginal_relevance_search](https://python.langchain.com/v0.2/api_reference/mongodb/vectorstores/langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch.html#langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch.max_marginal_relevance_search) and that would return the relevant slice of data, which in our case would be an entire paragraph. However, we can continue to harness the power of the LLM to [contextually compress](https://python.langchain.com/v0.2/api_reference/langchain/retrievers/langchain.retrievers.contextual_compression.ContextualCompressionRetriever.html#contextualcompressionretriever) the response so that it more directly tries to answer our question. 

```python
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Initialize MongoDB python client
client = MongoClient(params.MONGODB_CONN_STRING)
collection = client[params.DB_NAME][params.COLL_NAME]

# initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection, OpenAIEmbeddings(openai_api_key=params.OPENAI_API_KEY), index_name=params.INDEX_NAME
)
# perform a search between the embedding of the query and the embeddings of the documents
print("\nQuery Response:")
print("---------------")
docs = vectorStore.max_marginal_relevance_search(query, K=1)
#docs = vectorStore.similarity_search(query, K=1)

print(docs[0].metadata['title'])
print(docs[0].page_content)

# Contextual Compression
llm = OpenAI(openai_api_key=params.OPENAI_API_KEY, temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore.as_retriever()
)
```

```zsh
python3 query.py -q "Who started AT&T?"

Your question:
-------------
Who started AT&T?

AI Response:
-----------
AT&T - Wikipedia
"AT&T was founded as Bell Telephone Company by Alexander Graham Bell, Thomas Watson and Gardiner Greene Hubbard after Bell's patenting of the telephone in 1875."[25] "On December 30, 1899, AT&T acquired the assets of its parent American Bell Telephone, becoming the new parent company."[28]
```

## Resources
* [MongoDB Atlas](https://cloud.mongodb.com/)
* [Open AI API key](https://platform.openai.com/account/api-keys)
* [LangChain](https://python.langchain.com)
  * [WebBaseLoader](https://python.langchain.com/v0.2/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)
  * [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter)
  * [MongoDB Atlas module](https://python.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/)
  * [Contextual Compression. ](https://python.langchain.com/v0.2/api_reference/langchain/retrievers/langchain.retrievers.contextual_compression.ContextualCompressionRetriever.html#contextualcompressionretriever)
  * [MongoDBAtlasVectorSearch API](https://python.langchain.com/v0.2/api_reference/mongodb/index.html)


