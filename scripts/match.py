import os
import json
import numpy as np
from pymongo import MongoClient
import argparse

def cosine_similarity(a, b):
    """Compute cosine similarity between vectors"""
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_features(json_path):
    """Load features JSON"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)

def query_matches(features, uri, db_name="relove", collection_name="clothing", top_k=5, sim_threshold=0.7):
    """Query Atlas for similar dresses"""
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    
    matches = []
    input_vector = np.array(features["vector"])
    input_color = features["color"]
    input_style = features["style"]
    
    # Query dresses
    cursor = collection.find({"type": "dress"})
    for item in cursor:
        sim = cosine_similarity(input_vector, item["vector"])
        if sim >= sim_threshold:
            matches.append({
                "id": item["_id"],
                "image": item["image"],
                "color": item["color"],
                "type": item["type"],
                "style": item["style"],
                "similarity": float(sim)
            })
    
    # Sort and filter
    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
    filtered = []
    for m in matches:
        # Prefer same color or neutral
        if m["color"] == input_color or m["color"] == "neutral" or input_color == "neutral":
            filtered.append(m)
        # Fallback: Same style
        elif m["style"] == input_style:
            filtered.append(m)
    
    client.close()
    return filtered[:top_k]

def process_and_save(json_path, uri, output_dir="output", db_name="relove", collection_name="clothing"):
    """Match features, save results"""
    features = load_features(json_path)
    
    # Setup output dir
    matches_dir = os.path.join(output_dir, "matches")
    os.makedirs(matches_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(json_path))[0]
    
    # Query matches
    matches = query_matches(features, uri, db_name, collection_name)
    
    # Save JSON
    matches_path = os.path.join(matches_dir, f"{filename}_matches.json")
    with open(matches_path, "w") as f:
        json.dump(matches, f, indent=2)
    print(f"Saved matches: {matches_path}")
    
    return matches_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clothing Matching")
    parser.add_argument("--json_path", type=str, required=True, help="Features JSON path")
    parser.add_argument("--uri", type=str, required=True, help="MongoDB Atlas URI")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--db_name", type=str, default="relove", help="MongoDB database")
    parser.add_argument("--collection_name", type=str, default="clothing", help="MongoDB collection")
    
    args = parser.parse_args()
    
    try:
        matches_path = process_and_save(
            args.json_path,
            args.uri,
            args.output_dir,
            args.db_name,
            args.collection_name
        )
        print(f"Matching complete: {matches_path}")
    except Exception as e:
        print(f"Error: {str(e)}")