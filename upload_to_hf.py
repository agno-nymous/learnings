import json
import os
import base64
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_dataset(jsonl_path, base_path, limit=None):
    train_dir = os.path.join(base_path, 'datasets/welllog/train')
    eval_dir = os.path.join(base_path, 'datasets/welllog/eval')
    
    data = {'train': [], 'eval': []}
    
    with open(jsonl_path, 'r') as f:
        count = 0
        for line in f:
            if limit and count >= limit:
                break
            
            entry = json.loads(line)
            fname = entry['filename']
            
            # Identify split
            train_path = os.path.join(train_dir, fname)
            eval_path = os.path.join(eval_dir, fname)
            
            if os.path.exists(train_path):
                img_path = train_path
                split = 'train'
            elif os.path.exists(eval_path):
                img_path = eval_path
                split = 'eval'
            else:
                print(f"Warning: {fname} not found in train or eval dirs.")
                continue
                
            # Convert image to base64
            entry['image_base64'] = image_to_base64(img_path)
            data[split].append(entry)
            count += 1

    # Create HF Datasets
    ds_train = Dataset.from_list(data['train'])
    ds_eval = Dataset.from_list(data['eval'])
    
    return DatasetDict({'train': ds_train, 'eval': ds_eval})

if __name__ == "__main__":
    import huggingface_hub
    
    JSONL_PATH = "well_log_header.clean.jsonl"
    BASE_PATH = "/Users/ashishthomaschempolil/codefiles/learnings"
    
    # Try to load .env if python-dotenv is installed
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    TOKEN = os.environ.get("HF_TOKEN")
    if not TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables or .env file.")
    
    print("Logging in to Hugging Face...")
    huggingface_hub.login(token=TOKEN)
    api = huggingface_hub.HfApi()
    user_info = api.whoami()
    username = user_info['name']
    
    REPO_ID = f"{username}/well-log-headers-ocr"
    
    print("Preparing dataset (encoding images to base64)...")
    dataset = prepare_dataset(JSONL_PATH, BASE_PATH)
    
    print(f"Dataset prepared:")
    print(f"  Train: {len(dataset['train'])} items")
    print(f"  Eval: {len(dataset['eval'])} items")
    
    print(f"Pushing to Hugging Face: {REPO_ID}...")
    dataset.push_to_hub(REPO_ID, private=True)
    print(f"\nSuccess! Your dataset is available at: https://huggingface.co/datasets/{REPO_ID}")
