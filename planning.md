# WE ARE REFACTORING GPT-LAB
previous versions were very single python script on single machine with no abstraction, not modular, no tests, etc
the goal is to setup abstract base classes, a strict typing system, and a modular pipeline that is flexible to training not quite arbitrary models but somewhat close to it

### What needs to be abstract versus what limitations can we assume?
- *file storage* - assume local directory using git-lfs
### *SerDes*
- EVERYTHING has a `.save_to_dir(self, dir: str) -> None` and `.load_from_dir(cls, dir: str) -> cls`
- not only should the data be saved, but also the currently running python file as a .txt
    - when loading, we should check that the saved .txt backup is equal to the current file doing the loading
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
- take in a model with compatible `.forward` (or `.inference`?) types and output a metric
    - are metrics a custom datatype? 
    - do we save a given metric of all models to the same place in some leaderboard?
        - a shared storage place? do we save benchmark results with a model or with a benchmark?
- required methods:
    - `.`


# for DAGSeq2DAGSeq what we do is
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





























#
#
#
#
#
#

##

#
#
#
#
#
#

#
#
#
#

#
#
#
#
#

#
#
#
#

# BRB BATHROOM
