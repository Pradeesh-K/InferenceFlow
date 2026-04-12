# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.messages import HumanMessage, AIMessage
# from dotenv import load_dotenv
# load_dotenv()

# # Setup embedding model
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small", dimensions=768)

# # LLM Usage
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# # Fix the system prompt where i use variable
# prompt = (
# f"You are an customer Support Assistant for Amazon. You will get enquiries about status of the Package or general questions on Refund or Privacy
# For queries on status of package return the tracking id if available from user else ask them the tracking id
# For queries on Refund or Privacy use the information grounded based on context: {context}
# Answer format: {{
# "query_type": "status" or "general"
# "answer": "your answer"
# }}  "
# )


# # Order Database
# json_db = "package_data.json" # an array of   {
#                                             #     "tracking_id": "548",
#                                             #     "status": "Delayed or OnTime
#                                             #     "location": "Munich",
#                                             #     "expected_delivery_date": "2026-04-29"
#                                             # },

# #Embeddings containing Refund and Privacy Information
# embeddings_folder = "vector_store"
# embeddings_path = "vector_store/faiss_index.bin"
# text_chunk_path = "vector_store/chunks.txt"


# app.route():
#     user_query = request.get()
#     vector = embeddings.embed_document(user_query)
# # semantic caching: This vector is compared against existing query vectors - need to develop another faiss index at embeddings_folder named "faiss_query_index.bin" 
#     # if the query embedding is too similar to exising query index, we return the llm response
#     # we have to add system prompt to llm call
#     response = llm.invoke([HumanMessage(content=prompt)])
#     print(response.content)
#     # parse the answer json from response.content
#     answer_json = # result from parsing
#     if answer_json["query_type"] == "general" # improve the check:
#         # semantic caching before we send the answer, we have to do semantic caching , save the reponse in the vector_store folder in a seperate index
#         # thus any new query is compared to the existing queries embedded, and if query is too similar we can bypass the llm call
#         return answer_json["answer"]



import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
import faiss

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# --- Config ---
EMBEDDINGS_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
DIM = 768
SIMILARITY_THRESHOLD = 0.8
TOP_K = 3

# --- Paths ---
VECTOR_STORE_PATH = "vector_store"
CONTENT_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "faiss_index.bin")
CONTENT_CHUNKS_PATH = os.path.join(VECTOR_STORE_PATH, "chunks.txt")
QUERY_CACHE_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "faiss_query_index.bin")
QUERY_CACHE_TEXT_PATH = os.path.join(VECTOR_STORE_PATH, "query_cache.txt")
JSON_DB_PATH = "utils/package_data.json"

# --- Initialize models ---
embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL, dimensions=DIM)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# --- Load content FAISS + chunks ---
content_index = faiss.read_index(CONTENT_INDEX_PATH)
with open(CONTENT_CHUNKS_PATH, "r") as f:
    content_chunks = [line.strip() for line in f]

# --- Load or create query cache ---
if os.path.exists(QUERY_CACHE_INDEX_PATH):
    query_index = faiss.read_index(QUERY_CACHE_INDEX_PATH)
    with open(QUERY_CACHE_TEXT_PATH, "r") as f:
        query_cache = [line.strip() for line in f]
else:
    query_index = faiss.IndexFlatIP(DIM)  # cosine similarity
    query_cache = []

# --- Helper functions --- normalized query embedding
def embed_text(text):
    vec = np.array([embeddings.embed_query(text)]).astype("float32")
    faiss.normalize_L2(vec)   # ✅ normalize here
    return vec

def cosine_similarity(vec1, vec2):
    # normalize vectors for IP -> cosine
    vec1 = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    return np.dot(vec1, vec2.T)

def semantic_cache_check(query_vector):
    if len(query_cache) == 0:
        return None
    D, I = query_index.search(query_vector, k=1)
    if D[0][0] >= SIMILARITY_THRESHOLD:
        return query_cache[I[0][0]]
    return None

def get_context(query):
    # Embed query for retrieval
    q_vec = embed_text(query)
    D, I = content_index.search(q_vec, k=TOP_K)
    chunks = [content_chunks[i] for i in I[0]]
    return "\n".join(chunks)

# Query normalized so the cosine similarity is calculated during Inner product
def save_query_cache(query, response, q_vec):
    query_index.add(q_vec)  # already normalized
    query_cache.append(f"User: {query}\nBot: {response}")

    with open(QUERY_CACHE_TEXT_PATH, "w") as f:
        for line in query_cache:
            f.write(line + "\n")

    faiss.write_index(query_index, QUERY_CACHE_INDEX_PATH)

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query():
    user_query = request.json.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    # Check semantic cache first
    query_vec = embed_text(user_query)
    cached_response = semantic_cache_check(query_vec)
    if cached_response:
        return jsonify({"answer": cached_response})

    # Retrieve context from content index
    context = get_context(user_query)
    print(f"Context found: {context}")

    # Build system prompt
    prompt = f"""
    You are a customer Support Assistant for Amazon. You will get user enquiries about status of the Package or general questions on Refund or Privacy.
    For queries on status of package extract only the tracking id from the user query and use it as answer, if not found ask user the tracking id.
    For queries on Refund or Privacy use the information grounded based on context: {context}
    User Query: {user_query}
    Answer format: {{
    "query_type": "status" or "general",
    "answer": "your answer"
    }}
    """

    # Call LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    answer_text = response.content.strip()

    # Parse JSON safely
# --- Parse JSON safely and handle status queries ---
    try:
        answer_json = json.loads(answer_text)
    except:
        # fallback if parsing fails
        answer_json = {"query_type": "general", "answer": answer_text}

    # --- Handle tracking status lookup ---
    print(answer_json)
    if answer_json.get("query_type") == "status":
        tracking_id_str = answer_json.get("answer", "").strip()
        # try to parse as number
        try:
            tracking_id = str(int(tracking_id_str))
            # lookup in package_data.json
            with open(JSON_DB_PATH, "r") as f:
                package_data = json.load(f)
            match = next((p for p in package_data if p["tracking_id"] == tracking_id), None)
            if match:
                # return formatted string
                result = (f"Found tracking ID {match['tracking_id']}: "
                        f"Status: {match['status']}, "
                        f"Location: {match['location']}, "
                        f"Expected Delivery: {match['expected_delivery_date']}")
            else:
                result = "Tracking number not found, please provide the correct number."
        except ValueError:
            # not a number
            result = "Please provide your tracking ID so I can check your order status."
    else:
        # general query → return LLM answer
        result = answer_json.get("answer", "")

    # --- Save to semantic cache regardless ---
    save_query_cache(user_query, result, query_vec)

    return jsonify({"answer": result})

# --- Run ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
