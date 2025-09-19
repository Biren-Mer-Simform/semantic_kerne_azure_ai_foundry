import os
import json
import asyncio
from urllib.parse import quote_plus
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from pymongo import MongoClient
from pymongo.errors import OperationFailure

# Load environment variables
load_dotenv(".env", override=True)

def get_mongo_connection_string():
    mongo_connection_string = os.environ["AZURE_COSMOS_CONNECTION_STRING"]
    mongo_username = quote_plus(os.environ["AZURE_COSMOS_USERNAME"])
    mongo_password = quote_plus(os.environ["AZURE_COSMOS_PASSWORD"])
    return mongo_connection_string.replace("<user>", mongo_username).replace("<password>", mongo_password)

async def upsert_data_to_memory_store(client, data_file_path: str) -> None:
    """Insert/update data into Cosmos DB memory store from JSON file"""
    database = client.get_database(os.environ['AZURE_COSMOS_DATABASE_NAME'])
    collection = database.get_collection(os.environ['AZURE_COSMOS_COLLECTION_NAME'])
    print(f"Working with collection: {collection}")

    with open(file=data_file_path, encoding="utf-8") as f:
        data = json.load(f)
        
        for idx, item in enumerate(data, start=1):
            print(f"Processing item {idx}: {item.get('title', 'No title')}")
            
            new_document = {
                "_id": item.get('id'),
                "title": item.get('title'),
                "content": item.get('content'),
                "category": "movie"  # Adding category for filtering
            }
            
            # Check if document exists
            try:
                existing_doc = collection.find_one({"_id": item.get('id')})
                exists = existing_doc is not None
            except Exception as e:
                print(f"Error checking document existence: {e}")
                exists = False
            
            if not exists:
                try:
                    collection.insert_one(new_document)
                    print(f"✅ Inserted {idx}/{len(data)}: {item.get('title', 'No title')}")
                except Exception as e:
                    print(f"❌ Error inserting document {idx}: {e}")
            else:
                print(f"⏭️  Skipping {idx}/{len(data)} (already exists): {item.get('title', 'No title')}")

def setup_search_indexes(collection):
    """Setup appropriate search indexes for the collection"""
    try:
        # Get existing indexes
        existing_indexes = list(collection.list_indexes())
        print(f"Existing indexes: {[idx['name'] for idx in existing_indexes]}")
        
        # Check if we have a text index
        text_indexes = [idx for idx in existing_indexes if any(field[1] == 'text' for field in idx.get('key', {}).items())]
        
        if not text_indexes:
            # Create text index on multiple fields for better search
            collection.create_index([
                ('title', 'text'),
                ('content', 'text')
            ])
            print("✅ Created compound text index on title and content")
        else:
            print(f"📋 Text index already exists: {text_indexes[0]['name']}")
            
        # Create regular indexes for filtering
        try:
            collection.create_index([('category', 1)])
            print("✅ Created index on category")
        except OperationFailure:
            print("📋 Category index already exists")
            
        try:
            collection.create_index([('title', 1)])
            print("✅ Created index on title")
        except OperationFailure:
            print("📋 Title index already exists")
            
    except OperationFailure as e:
        if "ExactlyOneTextIndex" in str(e):
            print("⚠️  A text index already exists. Using existing text index for search.")
        else:
            print(f"❌ Error setting up indexes: {e}")

def search_documents(collection, query_term, limit=5):
    """Enhanced search function with multiple search strategies"""
    print(f"\n🔍 Searching for: '{query_term}'")
    
    # Strategy 1: Text search (if text index exists)
    try:
        text_results = list(collection.find(
            {"$text": {"$search": query_term}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit))
        
        if text_results:
            print(f"📝 Found {len(text_results)} results using text search")
            return text_results
    except OperationFailure as e:
        print(f"⚠️  Text search failed: {e}")
    
    # Strategy 2: Regex search on title and content
    regex_pattern = {"$regex": query_term, "$options": "i"}
    regex_results = list(collection.find({
        "$or": [
            {"title": regex_pattern},
            {"content": regex_pattern}
        ]
    }).limit(limit))
    
    if regex_results:
        print(f"🔤 Found {len(regex_results)} results using regex search")
        return regex_results
    
    # Strategy 3: Keyword-based search (split query and search for individual words)
    keywords = query_term.lower().split()
    keyword_conditions = []
    for keyword in keywords:
        keyword_regex = {"$regex": keyword, "$options": "i"}
        keyword_conditions.extend([
            {"title": keyword_regex},
            {"content": keyword_regex}
        ])
    
    if keyword_conditions:
        keyword_results = list(collection.find({
            "$or": keyword_conditions
        }).limit(limit))
        
        if keyword_results:
            print(f"🔑 Found {len(keyword_results)} results using keyword search")
            return keyword_results
    
    print("❌ No results found with any search strategy")
    return []

def display_search_results(results, query_term):
    """Display search results in a formatted way"""
    if not results:
        print(f"\n❌ No results found for '{query_term}'")
        return
    
    print(f"\n🎬 Search Results for '{query_term}':")
    print("=" * 60)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Title: {doc.get('title', 'No title')}")
        print(f"   ID: {doc.get('_id', 'No ID')}")
        
        content = doc.get('content', 'No content available')
        # Truncate content if too long
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"   Content: {content}")
        
        # Show relevance score if available
        if 'score' in doc:
            print(f"   Relevance Score: {doc['score']:.2f}")
        
        print("-" * 40)

async def main():
    mongo_connection_string = get_mongo_connection_string()
   
    # Initialize Kernel
    kernel = Kernel()

    # Add Azure OpenAI Chat Service
    kernel.add_service(
        AzureChatCompletion(
            service_id="chat_completion",
            deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
    )
    print("✅ Added Azure OpenAI Chat Service")

    # Add Azure OpenAI Embedding Service
    kernel.add_service(
        AzureTextEmbedding(
            service_id="text_embedding",
            deployment_name=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
    )
    print("✅ Added Azure OpenAI Embedding Service")

    # Creating mongo client
    client = MongoClient(mongo_connection_string)
    database = client.get_database(os.environ['AZURE_COSMOS_DATABASE_NAME'])
    collection = database.get_collection(os.environ['AZURE_COSMOS_COLLECTION_NAME'])
    
    # Setup search indexes
    print("🔧 Setting up search indexes...")
    setup_search_indexes(collection)
    
    # # Test different search queries
    # test_queries = [
    #     "godfather",
    #     "What do you know about the godfather?",
    #     "Ruth",
    #     "love",
    #     "movie"
    # ]
    
    # for query_term in test_queries:
    #     results = search_documents(collection, query_term, limit=3)
    #     display_search_results(results, query_term)
    #     print("\n" + "="*80 + "\n")
    
    # # Interactive search loop to query the DB
    # print("🎯 Interactive Search Mode (type 'quit' to exit):")
    # while True:
    #     query = input("\nEnter your search query: ").strip()
    #     if query.lower() in ['quit', 'exit', 'q']:
    #         break
        
    #     if query:
    #         results = search_documents(collection, query, limit=5)
    #         display_search_results(results, query)
    

if __name__ == "__main__":
    asyncio.run(main())