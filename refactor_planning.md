# WE ARE REFACTORING GPT-LAB
previous versions were very single python script on single machine with no abstraction, not modular, no tests, etc
the goal is to setup abstract base classes, a strict typing system, and a modular pipeline that is flexible to training not quite arbitrary models but somewhat close to it

# PLANNING
### What needs to be abstract versus what limitations can we assume?
- *file storage* - assume local directory using git-lfs
### *SerDes*
- EVERYTHING has a `.save_to_dir(self, dir: str) -> None` and `.load_from_dir(cls, dir: str) -> cls`
    - this generally calls recursively on all submodules
- also save the currently running root python file as a .txt inside this root `/experiments/YYYYMMDD_HHMMSS/` for replication/backup purposes
- when loading, we should assert that the saved .txt backup is equal to the current file doing the loading
- the experiments directory gets timestamped but if we want to reuse a given module frequently that's a manual copy & paste into the `/<module_name>/saved` directory?
- it's not reasonable to use git-lfs for the pytorch state-dict files. instead let's require each model implement a `.save_to_cloud()` and `.load_from_cloud()` which will probably end up using huggingface
### *LOGGING*
- a log file that goes inside each experiment subdirectory
- initialize a logger that then gets passed into each module upon initialization
- should support some derivative of the `print0` function that puts rank 0 to console but all ranks to log.txt
- 
### *DATA TYPES* 
- keep it abstract with strict typing so that we know what inputs & outputs each module accepts
- all modules specify datatype input & output for every single method they have
### *CONFIGS*
- force everyone to inherit from a strictly typed config
- technically they could get away with just saving & loading a dictionary but they'll likely follow the example
### *DATA SOURCES*
- needs to be able to:
    - get data from black-box source that user defines (eg. download from internet, load from pre-existing directory, etc)
    - OPTIONALLY pre-cache their data to local disc in an efficiently/quickly readable format
    - support multi-GPU? would you just pass one of pytorch's dataloaders in that case? **TODO**
- design/methods:
    - `.get_data_generator(self) -> Generator[CustomDataVal, None, None]:`
    - `.initialize(cls, cfg: DataSourceConfig) -> cls` 
        - eg. might require as an argument a model that is a tokenizer, meaning it has required inference type of
        `Union[Text, BatchedText]` and inference output `Union[TokenSequence, BatchedTokenSequence]`
    - for de-serialization on a data source that does pre-caching, we just save the path where pre-cached data lies, hope the data is still there, and if not call `.pre-cache()` again
### *MODELS*
- pretty abstract in order to support weird input/output types (eg. DAGSeq2DAGSeq, tokenizers count as models, etc)
- required methods: 
    - `.instantiate(cls, cfg: ModelConfig, logger: logger) -> cls:` creates a randomly initialized model
    - `.forward(self, input_dataloader: Generator[CustomDataVal, None, None]) -> Generator[CustomDataVal, None, None]:`
        - `.fwd_input_type(self) -> CustomDataType:` and `.fwd_output_type(self) -> CustomDataType`
    - `.inference(self, input_dataloader: Generator[CustomDataVal, None, None]) -> Generator[CustomDataVal, None, None]:`
        - `.inf_input_type(self) -> CustomDataType:` and `.inf_output_type(self) -> CustomDataType`
### *TRAINERS*
- take as input a model and a data source and outputs a new instance of that model class that has been trained, potentially while saving checkpoints to directories along the way, which themselves are instances of the trainer class
- required methods:
    - `.instantiate(cls, cfg: TrainerConfig, model: ModelCls, data_source: DataSourceCls) -> cls`
        - asserts in/out types of model.forward and data_source.get_data_generator match what it expects
    - `.load_checkpoint(cls, checkpoint_dir: str) -> cls`
        - because in the checkpont you saved info on the trainer, such as how many iterations had already gone through, that means to resume you'll have to run & not use the data generator for that many iterations, whereas the model checkpoint is a simpler load
    - `.train(self) -> ModelCls`
        - `.train_input_type(self) -> CustomDataType:` and `.train_output_type(self) -> CustomDataType`  
### *BENCHMARKS* 
- data for benchmarking must come in the form of a data source (eg. `HellaSwagDataSourceCls`)
- a benchmark module itself takes in a data source and a model and just handles the computation of the metric on the data (eg. `MultipleChoiceLogitsBenchmark`, `CorrectAnswerInTopK`, `SmarterLLMGrader`, etc)
- tbh we can just keep a helper method and some csv's somewhere for leaderboards
- no need to SerDes a specific instance of a benchmark; honestly everything can be static methods just organized or not even a class
- i suppose all benchmarks must be compatible with a model's `.forward` method? or optionally `.inference` method?
- required_methods:
    - `.calculate(self, data: DataSource, model: Model) -> CustomDataVal`
    - `.add_to_leaderboard`
- i feel like we have enough of the puzzle pieces ironed out that i can ignore benchmark design for now
### *TESTING*
- everyone's custom modules inherit from one of our ABCs and we first assert that and that they have the right abstractmethods
- test file loops through files in its directory using `__test__: bool` and `__test_name__: str` to let the tests know whether & what to test
- a given test file creates a mock data source, model, etc with the desired types from `.?_input_type()` and `.?_output_type()` which can likely be global fixtures
### *OTHER THOUGHTS*
- we'll create a `PyTorchModelCls` and a `PyTorchTrainerCls` to inherit and add any additional features specific to pytorch as well as enable further type assertions for module interaction compatibility
- `utils.py` handles a bunch of helper functions that are used throughout the codebase like `print0` and `import_from_nested_path` (are we still using that?)




## for DAGSeq2DAGSeq what we do is
1. download wikipedia dump, parse it, save each file to the same folder with titles as filenames and titles at beginning of doc
2. create DataSource object that for a __get_item__(self, idx=i) call reads off from the files using indexing from alphabetical order and once that file is read from disc it 
    1. tokenizes it and count number of tokens in this doc (and keeps track of total count)
    2. searches for special tokens `[`, `](`, and `)`
    3. grabs reference link, de-tokenizes it, and grabs the document from that reference link
    4. from all links at hand, uses custom graph traversal algorithm to decide which of them to read
    5. repeat steps i through iv until total number of tokens meets or surpasses max_seq_len
    6. concatenate all retrieved documents (which have already been tokenized) with <endoftext> token between them
    7. turn List[int] into np.ndarray of dtype uint and shape (max_seq_len,)
3. save each of those np.ndarrays into their own bin file? or back-to-back in big bin files? 
    - the former would ensure a continuous chosen graph in a single batch, which means more "complete" graphs to be trained on less data diversity, and potentially some overhead from beginning reads on new bin files
4. the `.get_data_generator()` method reads from bin files to provide the next iteration
5. then inside the model's `.forward()` we do `torch.cumsum()` operations similar to how `<|endoftext|>` ones work for block-causal masks in order to create the flex-attention mask
