import json
import random
from datetime import datetime, timedelta

# Config
NUM_RECORDS = 500
tracking_ids = random.sample(range(100, 1000), NUM_RECORDS)
CITIES = ["Munich", "Berlin", "Hamburg", "Frankfurt", "Stuttgart", "Cologne"]
STATUSES = ["OnTime", "Delayed"]

start_date = datetime(2026, 4, 11)
end_date = datetime(2026, 4, 30)
date_range = (end_date - start_date).days

data = []

for tracking_id in tracking_ids:
    record = {
        "tracking_id": str(tracking_id),
        "status": random.choice(STATUSES),
        "location": random.choice(CITIES),
        "expected_delivery_date": (
            start_date + timedelta(days=random.randint(0, date_range))
        ).strftime("%Y-%m-%d")
    }
    data.append(record)

# Write to JSON file in root folder
with open("package_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Generated package_data.json with 900 records.")