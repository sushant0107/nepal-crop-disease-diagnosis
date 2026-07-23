import os, zipfile
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "w4ashabii/nepali_crop_data"
EXTRACT_TO = "/content/training_data"
os.makedirs(EXTRACT_TO, exist_ok=True)

# list zip parts in repo
api = HfApi()
files = api.list_repo_files(REPO_ID, repo_type="dataset")
zip_files = sorted(f for f in files if f.startswith("zips/") and f.endswith(".zip"))
print("Found zips:", zip_files)

# download and extract each
for zf_name in zip_files:
    local_zip = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=zf_name,
    )
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(EXTRACT_TO)
    print("Extracted", zf_name)

print("Training data ready at:", EXTRACT_TO)
