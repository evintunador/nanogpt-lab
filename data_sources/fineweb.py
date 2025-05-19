"""
API MUST CONTAIN:
get_dataset(dataset_config, split, seed) -> datasets.Dataset:

and the datasets.Dataset should be useable like
item = next(dataset)['text']
for getting the raw text, with other optional dict keys
being for metadata
"""
import os
import random as r

from datasets import load_dataset

# set __test__ to False for any file in tokenizers/ that should not be tested
# (AKA purposely does not meet API requierments)
__test__ = True

"""
FineWeb dataset
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""
default_cfg = {
    'streaming': True,
    'shuffle': False,
    'edu': False
}

def get_dataset(
        cfg: dict, 
        split: str = 'train',
        seed: int = r.randint(0, 2**32 - 1)
    ):
    cfg = {**default_cfg, **cfg}
    fw = load_dataset(
        "HuggingFaceFW/fineweb" + ("-edu" if cfg['edu'] else ""), 
        name=f"sample-350BT", 
        split=split, 
        streaming=cfg['streaming'],
    )
    if cfg['shuffle']: fw = fw.shuffle(seed=seed)
    return fw

