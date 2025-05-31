import random as r

from datasets import load_dataset, Dataset

from data_sources.datasource import DataSourceConfig, DataSource

# set __test__ to False for any file in tokenizers/ that should not be tested
# (AKA purposely does not meet API requierments)
__test__ = True
# info the automated tests need
__config_name__ = "FinewebConfig"
__datasource_name__ = "Fineweb"


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


class FinewebConfig(DataSourceConfig):
    def __init__(
            self,
            filename: str,
            edu: bool = False,
            split: str = 'train',
            sample: str = "sample-350BT",
            streaming: bool = True,
            shuffle: bool = False,
            seed: int = r.randint(0, 2*32 - 1),
    ):
        super().__init__(filename=filename)
        self._edu = edu
        self._split = split
        self._sample = sample
        self._streaming = streaming
        self._shuffle = shuffle
        self._seed = seed

    @property
    def edu(self):
        return self._edu

    @property
    def split(self):
        return self._split

    @property
    def sample(self):
        return self._sample

    @property
    def streaming(self):
        return self._streaming

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def seed(self):
        return self._seed


class  Fineweb(DataSource):
    @staticmethod
    def get_datasource(cls, config: FinewebConfig) -> Fineweb:
        fw = load_dataset(
            "HuggingFaceFW/fineweb" + ("-edu" if config.edu else ""),
            name=config.sample,
            split=config.split,
            streaming=config.streaming,
        )
        if config['shuffle']: fw = fw.shuffle(seed=seed)
        return Fineweb(config=config, hg_dataset=fw)

    def __init__(self, config: FinewebConfig, hg_dataset: Dataset):
        super().__init__(config=config)
        self._hg_dataset = hg_dataset

    def __getitem__(self, key):
        return self._hg_dataset[key]['text']

    def __len__(self):
        return len(self._hg_dataset)

    def __iter__(self):
        return iter(self._hg_dataset)
