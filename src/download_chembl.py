# import requests

# TARGET_ID = "CHEMBL4822"

# url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

# params = {
#     "target_chembl_id": TARGET_ID,
#     "limit": 5,
# }

# response = requests.get(url, params=params)

# print("Status code:", response.status_code)

# data = response.json()

# print("Number of records returned:", len(data["activities"]))

# print("\nFirst record:")
# print(data["activities"][0])
import requests
import pandas as pd

# ==========================================
# BACE1 TARGET
# ==========================================

TARGET_ID = "CHEMBL4822"

URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

# ==========================================
# DOWNLOAD SETTINGS
# ==========================================

LIMIT = 1000
OFFSET = 0

params = {
    "target_chembl_id": TARGET_ID,
    "limit": LIMIT,
    "offset": OFFSET
}

# ==========================================
# REQUEST DATA
# ==========================================

print("Downloading BACE1 activity data...")

response = requests.get(URL, params=params)

print("Status code:", response.status_code)

response.raise_for_status()

data = response.json()

activities = data["activities"]

print("Records downloaded:", len(activities))

# ==========================================
# CONVERT TO DATAFRAME
# ==========================================

df = pd.DataFrame(activities)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ==========================================
# SAVE RAW DATA
# ==========================================

output_file = "data/raw/bace1_activity_raw.csv"

df.to_csv(output_file, index=False)

print("\nSaved dataset to:")
print(output_file)