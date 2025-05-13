"""
API MUST CONTAIN:
get_dataset(dataset_config, split, seed) -> datasets.Dataset:

and the datasets.Dataset should be useable like
item = next(dataset)['text']
for getting the raw text, with other optional dict keys
being for metadata
"""