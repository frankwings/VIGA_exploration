import os
print(f"HF_TOKEN env: {os.environ.get('HF_TOKEN', 'Not Set')}")
print(f"HUGGING_FACE_HUB_TOKEN env: {os.environ.get('HUGGING_FACE_HUB_TOKEN', 'Not Set')}")

try:
    from huggingface_hub import hf_hub_download
    print("Attempt 1: microsoft/TRELLIS.2-4B (pipeline.json)")
    path = hf_hub_download(repo_id='microsoft/TRELLIS.2-4B', filename='pipeline.json')
    print(f"Success: {path}")
    with open(path, 'r') as f:
        print("PIPELINE_JSON_START")
        print(f.read())
        print("PIPELINE_JSON_END")
except Exception as e:
    print(f"Failed Attempt 1: {e}")

try:
    from huggingface_hub import list_repo_files
    print("Listing files in microsoft/TRELLIS.2-4B:")
    files = list_repo_files("microsoft/TRELLIS.2-4B")
    for f in files:
        print(f)
except Exception as e:
    print(f"Failed to list files: {e}")

