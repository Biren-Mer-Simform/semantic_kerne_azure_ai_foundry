# load the environment variables file
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
import asyncio

import json

# from semantic_kernel.memory.memory_store_base import MemoryStoreBase
# from semantic_kernel.connectors.memory_stores.azure_cosmosdb import (
#     AzureCosmosDBMemoryStore,
# )
# or if using the MongoDB variant
from semantic_kernel.connectors.azure_cosmos_db import CosmosMongoStore

from semantic_kernel.connectors.in_memory import InMemoryCollection
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)

from semantic_kernel.connectors.memory_stores.azure_cosmosdb.azure_cosmos_db_memory_store import (
    AzureCosmosDBMemoryStore,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType
)

from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory


load_dotenv(".env", override=True)

# Read and Store Environment variables
def get_mongo_connection_string():
    mongo_connection_string = os.getenv("AZURE_COSMOS_CONNECTION_STRING", "<YOUR-COSMOS-DB-CONNECTION-STRING>")
    mongo_username = quote_plus(os.getenv("AZURE_COSMOS_USERNAME"))
    mongo_password = quote_plus(os.getenv("AZURE_COSMOS_PASSWORD"))
    return mongo_connection_string.replace("<user>", mongo_username).replace("<password>", mongo_password)




async def upsert_data_to_memory_store(memory: SemanticTextMemory, store: InMemoryCollection, data_file_path: str) -> None:
    """
    This asynchronous function takes two memory stores and a data file path as arguments.
    It is designed to upsert (update or insert) data into the memory stores from the data file.

    Args:
        memory (callable): A callable object that represents the semantic kernel memory.
        store (callable): A callable object that represents the memory store where data will be upserted.
        data_file_path (str): The path to the data file that contains the data to be upserted.

    Returns:
        None. The function performs an operation that modifies the memory stores in-place.
    """
    # collection name will be used multiple times in the code so we store it in a variable
    mongo_connection_string = get_mongo_connection_string()
    database_name = os.getenv("AZURE_COSMOS_DATABASE_NAME")
    collection_name = os.getenv("AZURE_COSMOS_COLLECTION_NAME")

    # Vector search index parameters
    index_name = os.getenv("AZURE_COSMOS_INDEX_NAME", "VectorSearchIndex")
    vector_dimensions = 1536  # text-embedding-ada-002 uses a 1536-dimensional embedding vector
    num_lists = 100
    similarity = CosmosDBSimilarityType.COS
    kind = CosmosDBVectorSearchType.VECTOR_IVF
    m = 16
    ef_construction = 64
    ef_search = 40

    with open(file=data_file_path, encoding="utf-8") as f:
        data = json.load(f)
        n = 0
        for item in data:
            n += 1
            # check if the item already exists in the memory store
            # if the id doesn't exist, it throws an exception
            try:
                already_created = bool(await store.get(collection_name, item["id"], with_embedding=True))
            except Exception:
                already_created = False
            # if the record doesn't exist, we generate embeddings and save it to the database
            if not already_created:
                await memory.save_information(
                    collection=os.environ["AZURE_COSMOS_COLLECTION_NAME"],
                    id=item["id"],
                    # the embedding is generated from the text field
                    text=item["content"],
                    description=item["title"],
                )
                print(
                    "Generating embeddings and saving new item:",
                    n,
                    "/",
                    len(data),
                    end="\r",
                )
            else:
                print("Skipping item already exits:", n, "/", len(data), end="\r")

async def main():

    # collection name will be used multiple times in the code so we store it in a variable
    mongo_connection_string = get_mongo_connection_string()
    database_name = os.getenv("AZURE_COSMOS_DATABASE_NAME")
    collection_name = os.getenv("AZURE_COSMOS_COLLECTION_NAME")

    # Vector search index parameters
    index_name = os.getenv("AZURE_COSMOS_INDEX_NAME", "VectorSearchIndex")
    vector_dimensions = 1536  # text-embedding-ada-002 uses a 1536-dimensional embedding vector
    num_lists = 100
    similarity = CosmosDBSimilarityType.COS
    kind = CosmosDBVectorSearchType.VECTOR_IVF
    m = 16
    ef_construction = 64
    ef_search = 40
    

    # Initialize the kernel
    kernel = Kernel()

    # adding azure openai chat service
    chat_model_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    kernel.add_service(
        AzureChatCompletion(
            service_id="chat_completion",
            deployment_name="gpt-4.1-mini",
            endpoint="https://chatbot-exp.cognitiveservices.azure.com/",
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
    )
    print("Added Azure OpenAI Chat Service...")

    # adding azure openai text embedding service
    embedding_model_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

    kernel.add_service(
        AzureTextEmbedding(
            service_id="text_embedding",
            endpoint="https://chatbot-exp.cognitiveservices.azure.com/",
            deployment_name="text-embedding-3-small",
            api_key=os.environ['AZURE_OPENAI_API_KEY']
        )
    )
    print("Added Azure OpenAI Embedding Generation Service...")

    print("Creating or updating Azure Cosmos DB Memory Store...")
    # create azure cosmos db for mongo db vcore api store and collection with vector ivf
    # currently, semantic kernel only supports the ivf vector kind
    store = await AzureCosmosDBMemoryStore.create(
        cosmos_connstr=mongo_connection_string,
        cosmos_api="mongo-vcore",
        database_name=database_name,
        collection_name=collection_name,
        index_name=index_name,
        vector_dimensions=vector_dimensions,
        num_lists=num_lists,
        similarity=similarity,
        kind=kind,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    
    store = CosmosMongoStore(
        connection_string=mongo_connection_string,
        database_name=database_name,
    )

    # Get (or create) a vector collection
    collection = await store.(
        name=collection_name,
        dimensions=vector_dimensions,
        similarity_metric=similarity,   # e.g., "cosine", "euclidean"
        index_kind=kind,                # e.g., "ivf", "hnsw"
        num_lists=num_lists,            # IVF parameter
        m=m,                            # HNSW parameter
        ef_construction=ef_construction,
        ef_search=ef_search,
    )


    print("Finished updating Azure Cosmos DB Memory Store...")


    memory = SemanticTextMemory(storage=store, embeddings_generator=kernel.get_service("text_embedding"))
    kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPluginACDB")
    print("Registered Azure Cosmos DB Memory Store...")
    
    # cleaned-top-movies-chunked.json contains the top 344 movie from the IMDB movies dataset
    # You can also try the text-sample.json which contains 107 Azure Service.
    # Replace the file name cleaned-top-movies-chunked.json with text-sample.json

    print("Upserting data to Azure Cosmos DB Memory Store...")
    await upsert_data_to_memory_store(memory, store, "./src/data/cleaned-top-movies-chunked.json")
    # each time it calls the embedding model to generate embeddings from your query
    query_term = "What do you know about the godfather?"
    result = await memory.search(collection_name, query_term)
    print(
        f"Result is: {result[0].text}\nRelevance Score: {result[0].relevance}\nFull Record: {result[0].additional_metadata}"
    )
if __name__ == "__main__":
    asyncio.run(main())