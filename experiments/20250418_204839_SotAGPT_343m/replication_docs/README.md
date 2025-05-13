# GPT-Lab **(currently in ALPHA)**
this repo is a massive overhaul of [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) with the goal of being a base for amateurs to do cheap & easy LLM experiments at a large enough scale to be worthy of an arxiv preprint. the idea is that repos like Modded-NanoGPT, [NanoGPT](https://github.com/karpathy/nanoGPT), [TinyLlama](https://github.com/jzhang38/TinyLlama), [picotron](https://github.com/huggingface/picotron?tab=readme-ov-file), and [Meta's Lingua](https://github.com/facebookresearch/lingua), are either too old of an architecture, too purpose-specific, not from-scratch enough, too expensive to run, too overly-complicated, not well setup for quickly iterating research ideas, etc and we plan to occupy a unique balance of those trade-offs

**this repo is currently in alpha, meaning that I think it's somewhat workable but have not utilized it on enough of my own experiments to guarantee that. before taking it out of alpha I will:**
    
1. implement the further improvements defined in the todo section below and 
2. go and implement a few experiment ideas and use what I learn from the difficulties I run into to add more things to the todo list

check out the video I made about it:

[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/4cvBgHMDISs/0.jpg)](https://www.youtube.com/watch?v=4cvBgHMDISs)

## getting started
the input arguments in these instructions are comically small values designed to get you up and running on the tiniest GPU(s) for demonstration purposes; in practice you'll have to tune them to properly utilize the available VRAM of your setup

1. either have one or more GPUs or hook up to a cloud GPU. for the latter see [this tutorial](https://youtu.be/mmRlZKFLAvE); i recommend [vast.ai](vast.ai) since they're always at or near the cheapest
2. either fork or create a template of this repo
3. `pip install -r requirements.txt`
4. train your tokenizer on fineweb. samples is the number of text characters to train on (split up evenly across all GPUs). vocabulary size should exclude any special tokens you plan on using later. for a tutorial on how Byte-Pair Encoding (BPE) tokenizers work, see [andrej karpathy's video](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s) for a simple & slow CPU implementation

single GPU:
```
python train_tokenizer.py --samples 100000 --vocabsize 1000 --name readmetokenizer --demo
```
multiple GPUs (replace `G` with the number of GPUs you have):
```
torchrun --nproc_per_node=G train_tokenizer.py --samples 100000 --vocabsize 1000 --name readmetokenizer --demo
```
5. download the [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset and convert all the raw text into tokens. dataset options are 10B, 100B, 10Bedu (default), or 100Bedu. tune shard_size (default 100 million) and num_shards to the quantity of data for your desired training run length. the script will only create one shard for the validation set which is not included in the count of num_shards
```
python download_fineweb.py --num_shards 1 --version 10B --shard_size 10000000 --tokenizer readmetokenizer_v1000_n100000.pkl
```
6. download the hellaswag benchmark: `python download_hellaswag.py`
7. train your language model. vocabulary size must be equial to your tokenizer size PLUS any special tokens defined in this script (1 for '<|endoftext|>', so 1000 + 1 = 10001). **WARNING:** if you include `--save_model` that will create a `.pt` file of the model weights, but by default the `.gitignore` will now allow this file to be pushed to github with the rest of the repo. this is done because the filesize is too large for github, and it means you have to find a way to download the model weights manually if you're on a cloud GPU and want to keep them

single GPU:
```
python train_gpt.py --model_name ReadmeGPT --tokenizer readmetokenizer_v1000_n100000.pkl --vocab_size 1001 --model_dim 128 --num_heads 4 --num_layers 6
```
multiple GPUs (replace `G`):
```
torchrun --nproc_per_node=G train_gpt.py --model_name ReadmeGPT --tokenizer readmetokenizer_v1000_n100000.pkl --vocab_size 1001 --model_dim 128 --num_heads 4 --num_layers 6
```
8. look in `experiments/` for your model. you should see 1) a `.txt` backup of all the `.py` files we just ran at the time of training (except `train_tokenizer.py`, which is backed inside the tokenizer `.pkl` file and therefore not readable from a file browser), 2) a `.csv` containing the training time & loss, 3) a log file containing important information such as the hellaswag benchmark score and the maximum memory allocated during training, and 4) maybe a `.pt` file if you elected to run with `--save_model`
9. great, now that all that is confirmed to be up & working you can start editing the code and running your own experiments by building off the baselines below!

## baselines

we've trained some baselines for your experiments to compare against. For now (while the repo is in alpha/beta), they are absurdly sh\*tty and really only here for demonstration purposes. As the repo improves, we will push new improved baselines of larger sizes, with better tuned hyperparameters, trained on more tokens, etc. The goal is to eventually closely resemble the GPT2 series of models in parameter count (maybe even larger) and train on as many tokens as possible while still keeping costs realistic for dedicated amateurs
| Baseline             | XS                | S                | M                |
| :------------------ | :-------------------------- | :--------------------------- | :--------------------------- |
| **Parameters (millions)**      | 57.8                       | 117.7                       | 342.5                       |
| **Tokens Traind On (billions)**      | 0.1                       | 0.4                       | 1.0                       |
| **GPU**             | RTX 3070                    | RTX 4060 Ti                  | A40                          |
| **VRAM Per GPU**  | 8GB                         | 16GB                         | 45GB                         |
| **GPU Count**       | 1                           | 2                            | 4                            |
| **GPU Cost  Per Hour (US Dollars)** | $0.113                      | $0.257                       | $1.761                       |
| **Trainimg Time (minutes)**| 12.02                       | 51.59                        | 94.93                       |
| ***Estimated* Total Cost (US Dollars)**  | $0.14                       | $0.48                        | $4.55                        |

*NOTES:* 
- Total cost is estimated as `Total Cost = ((Training Time) + (60 minutes)) * (GPU Cost  Per Hour)` to reflect the overhead of starting up your cloud GPU instance, testing which hyperparameters best utilize VRAM, running validation data and benchmarks, pushing changes and closing down your instance.
- All costs reflect GPUs rented from [vast.ai](vast.ai) on Apr 18, 2025

## todos / planned features:
- [ ] refactor to make everything modular & clarify what parts of the codebase can & cannot be manipulated for an experiment
    - [ ] build some kind of (semi-)automated testing framework to check each PR for bugs
    - [ ] create template documents to guide people in adding new modules, models, benchmarks, tokenizers, etc
    - [ ] split out config into its own .yaml file or something
- [ ] implement (optional) pipeline parallelism
- [ ] implement (optional) tensor parallelism
- meta
    - [ ] write a `how_to_experiment.md` to detail best practices for people looking to conduct scientifically robust experiments
    - [ ] write a `how_to_contribute.md` to detail best practices for potential non-model code contributions (bug fixes, minor obvious improvements)
- [ ] excessively comment and explain everything that's happening in each file
    - [ ] tensor shapes for every operation
    - [ ] ensure consistency in comment style (eg. choose between (B,N,D) and (batch_size, seq_len, model_dim))
- [ ] **implement more of my ideas using the code as it stands as a baseline to test & learn more about how this repo should work**
- `train_tokenizer.py`
    - [ ] change dataloader to load small chunks of dataset at a time from SSD to CPU-RAM to VRAM; just now I trained `tokenizers/gpt4regex_v50256_n1000000000.pkl` on 8x4070Ti's but only filled up 6-7GB of their available 16GB since for a dataset any larger the CPU would run out of memory before I could even get data onto the GPU
    - [ ] switch the backup/logging to be more `train_gpt.py` style. keep the `.pkl` for actual use (storage of regex pattern & merges) but save it in a folder alongside a `.txt` backup of `train_tokenizer.py` and another `.txt` file to list out & visualize all the merges
        - [ ] make corresponding changes inside `train_gpt.py`
- `train_gpt.py`
    - **planned** architecture edits (*if* they speed up / improve performance)
        - [x] adjust value embeddings to dynamically account for any number of layers to be either a function of model size, learnable, or something else that makes more sense
        - [ ] change values originally over-optimized for GPT2-124m 
            - [x] attention head scaling factor
            - [ ] are there any more?
        - [ ] re-implement Modded-NanoGPT's original attention masks (see [`def create_blockmasks()`](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py)
            - [ ] alternate between full-causal and sliding-window attention
                - [ ] make full-sliding pattern dynamically account for different numbers of model layers (similar to description of value embeddings above)
            - [ ] gradually increase window size as a function of training steps
        - [ ] user Liger kernel's fused CE Loss (or are we only using pytorch for this file and splitting off a separate version that's allowed to use custom kernels? idk) which would require either making our own custom version or getting rid of the scaling in-between the logits & the CE Loss
        - [ ] implement [mu-parameterization](https://github.com/EleutherAI/nanoGPT-mup)
    - **potential** architecture edits (*if* they speed up / improve performance)
        - [ ] go back and rapidly test a bunch of boring architecture edits (eg. MLP activation functions) to see whether those chosen by Modded-NanoGPT were really just over-fitting their dataset
        - [ ] MLA?
- beginner friendly versions for those who want to work off of a more well known architecture as a base (this would be more expensive due to slower training times)
    - [ ] `nanogpt.py`
        - [ ] optionally use downloadable actual GPT2 tokenizer or our own that uses GPT2's regex
    - [ ] `llama3.py`
        - [ ] there are a lot of missing details that were never released about how Llama3 was trained (such as dropout locations, optimizer, learning rates, etc) that we should fill in with a compromise between efficient methods from `train_gpt.py` and methods that are likely to be easily understandable for someone looking to wor with a simpler repo (eg. don't use Muon)
        - [ ] download llama's tokenizer
- [x] train models on 1x8GB vram, 2x16 GB, 4x32GB, and 8x80GB (for how much data each??) and record how much $ each one cost to run so that people have an estimate before doing their experiments
    - [ ] use chinchilla-optimal model size & data quantity
    - [x] set hyperparameter defaults to that of 1x8GB version
- more/improving benchmarks:
    - [ ] add batched inference support and then use it to speed up hellaswag benchmark
    - [ ] figure out what additional benchmarks make sense for models of this scale
    - [ ] api calls to a smarter LLM judge for mass comparisons of generated outputs?
        - [ ] create list of prompts (preferably from some pre-existing well vetted benchmark)
        - [ ] run model on said prompts right after the hellaswag benchmark
        - [ ] save outputs in some easily parseable format
        - [ ] write a script to stay in the root directory that
            1. takes in the file names of two different models as input (one baseline and one experiment)
            2. uploads outputs to some smarter LLM API (OpenAI, Anthropic, etc.) and has them pick which of the two outputs is better
            3. returns a win-rate (50% is random-chance) & confidence interval and saves it to the experiment model's directory
        - [ ] preferably after we do this once or twice i'd like to record an estimate of how much $ it takes each time so that people know
        - [ ] optionally instead of the outputs of this prompt being gathered & saved immediately after training we could have this only work for runs that called `--save_model` and therefore have an available .pt file to use, but that'd require storing the .pt files of the baselines somewhere
- [ ] write latex preprint skeleton for others to begin theirs from
    - [ ] specifics of the Modded-NanoGPT architecture in an appendix
    - [ ] write a few scripts that take experiment directories as input and output loss plot, benchmark tables, etc
- [ ] implement some sort of SFT/RLHF/DPO/RL as an option for the largest scale of model once the field settles on a technique? I don't think this makes sense given that even if our model size is big enough the odds that anybody is training on enough data for a model to be useable is pretty low for the time being