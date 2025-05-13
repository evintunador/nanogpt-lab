# python built-ins
import yaml
# pip installs
# local imports
from utils import import_from_nested_path

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
dataset_config = config.get('dataset', {})
tokenizer_config = config.get('tokenizer', {})

get_dataloader = import_from_nested_path(
    nested_folders=['datasets'],
    filename=dataset_config['name'],
    items=['get_dataloader']
)
_ = import_from_nested_path(
    nested_folders=['tokenizers'], 
    filnamee=tokenizer_config['name'], 
    items=['train_tokenizer', 'demo_tokenizer', 'save_tokenizer', 'load_tokenizer']
)
train_tokenizer, demo_tokenizer, save_tokenizer, load_tokenizer = _

dataloader = get_dataloader(dataset_config)
tokenizer = train_tokenizer(dataloader, tokenizer_config)
save_tokenizer(tokenizer, tokenizer_config)

tokenizer = load_tokenizer(tokenizer_config)
demo_text = "The quick brown fox jumps over the lazy dog."
assert tokenizer.dec(tokenizer.enc(demo_text)) == demo_text
demo_tokenizer(tokenizer, demo_text)

# TODO: logs for replications and whatever, save to same folder as tokenizer