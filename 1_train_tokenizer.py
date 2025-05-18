import os
import yaml
import pickle

from utils import import_from_nested_path

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
dataset_config = config.get('dataset', {})
tokenizer_config = config.get('tokenizer', {})

imported_dataset_items = import_from_nested_path(
    nested_folders=['custom_datasets'],
    filename=dataset_config['filename'],
    items=['get_dataset']
)
get_dataset = imported_dataset_items['get_dataset']

imported_tokenizer_items = import_from_nested_path(
    nested_folders=['tokenizers'], 
    filename=tokenizer_config['filename'], 
    items=['train_tokenizer', 'demo_tokenizer', 'load_tokenizer']
)
train_tokenizer = imported_tokenizer_items['train_tokenizer']
demo_tokenizer = imported_tokenizer_items['demo_tokenizer']
load_tokenizer = imported_tokenizer_items['load_tokenizer']

dataloader = get_dataset(dataset_config)
tokenizer = train_tokenizer(dataloader, tokenizer_config)

def save_tokenizer(filename: str, nickname: str)
    save_dir = os.path.join(
        os.path.dirname(__file__), 
        f"tokenizers/trained/{filename}_{nickname}"
    )
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"tokenizer.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {filepath}")
save_tokenizer(
    filename=tokenizer_config['filename'],
    nickname=tokenizer_config['nickname']
)

tokenizer = load_tokenizer(save_dir)
demo_text = "The quick brown fox jumps over the lazy dog."
assert tokenizer.dec(tokenizer.enc(demo_text)) == demo_text
demo_tokenizer(tokenizer, demo_text)

# TODO: logs for replications and whatever, save to same folder as tokenizer