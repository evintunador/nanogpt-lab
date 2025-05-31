import os
import yaml
import pickle

from utils import import_from_nested_path, visualise_tokens

# open configs
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
dataset_cfg = config.get('dataset', {})
tok_cfg = config.get('tokenizer', {})

# fetch custom dataset
imported_dataset_items = import_from_nested_path(
    nested_folders=['data'],
    filename=dataset_cfg['filename'],
    items=['get_dataset']
)
get_dataset = imported_dataset_items['get_dataset']
dataset = get_dataset(dataset_cfg)

# fetch and train custom tokenizer
imported_tokenizer_items = import_from_nested_path(
    nested_folders=['tokenizers'], 
    filename=tok_cfg['filename'], 
    items=['Tokenizer']
)
Tokenizer = imported_tokenizer_items['Tokenizer']
tokenizer = Tokenizer.train(dataset, tok_cfg)

# save the newly trained tokenizer
save_dir = os.path.join(
    os.path.dirname(__file__), 
    f"tokenizers/trained/{tok_cfg['filename']}_{tok_cfg['nickname']}"
)
os.makedirs(save_dir, exist_ok=True)
filepath = os.path.join(save_dir, f"tokenizer.pkl")
with open(filepath, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved to {filepath}")

# test and demo the tokenizer
demo_text = "The quick brown fox jumps over the lazy dog."
assert tokenizer.dec(tokenizer.enc(demo_text)) == demo_text
demo_bytes = demo_text.encode("utf-8")
assert tokenizer.dec_bytes(tokenizer.enc_bytes(demo_bytes)) == demo_bytes
visualise_tokens([tokenizer.dec_bytes([b]).encode("utf-8", errors="replace") 
                  if isinstance(b, int) else b 
                  for b in tokenizer.enc_bytes(demo_bytes)])

# TODO: logs for replications and whatever, save to same folder as tokenizer