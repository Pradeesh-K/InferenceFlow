import requests
import time
import random

queries = [
    "track order 123",
    "refund policy for watches?",
    "where is my package?",
    "what is return policy of sim card?",
    "refund policy for watches",
    "can i return a luxury handbag",

]

for _ in range(10):
    q = random.choice(queries)
    requests.post("http://localhost:8000/query", json={"query": q})
    print(f"query {q} submitted")
    time.sleep(random.uniform(0.2, 2))