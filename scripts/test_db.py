from pymongo import MongoClient

uri = "mongodb+srv://relove_user:myrelovepass007@relovecluster.3efxwzq.mongodb.net/?retryWrites=true&w=majority&appName=ReloveCluster"
client = MongoClient(uri)
db = client["relove"]
collection = db["clothing"]
print(f"Items: {collection.count_documents({})}")
for item in collection.find():
    print(item["_id"])
client.close()