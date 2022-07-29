# Incorporating Centering Theory into Neural Coreference Resolution

This repository contains the code and trained model from the paper ["Incorporating Centering Theory into Neural Coreference Resolution"](https://aclanthology.org/2022.naacl-main.218.pdf).


## Set up

#### Requirements
Set up a virtual environment with Python 3.7 and run: 
```
pip install -r requirements.txt
```

Follow the [Quick Start](https://github.com/NVIDIA/apex) to enable mixed precision using apex.

#### Prepare the dataset

This repo assumes access to the [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus.
Convert the original dataset into jsonlines format using:
```
export DATA_DIR=<data_dir>
python minimize.py $DATA_DIR
```

## Evaluation
Download our [trained model](https://drive.google.com/file/d/1vjNBFBAGVOJ0Pyy6A-NX3gaOswjCIE6X/view?usp=sharing).

and run:
```
export OUTPUT_DIR=<output_dir>
export CACHE_DIR=<cache_dir>
export MODEL_DIR=<model_dir>
export DATA_DIR=<data_dir>
export SPLIT_FOR_EVAL=<dev or test>

python run_coref.py \
        --output_dir=$OUTPUT_DIR \
        --cache_dir=$CACHE_DIR \
        --model_type=longformer \
        --model_name_or_path=$MODEL_DIR \
        --tokenizer_name=allenai/longformer-large-4096 \
        --config_name=allenai/longformer-large-4096  \
        --train_file=$DATA_DIR/train.english.jsonlines \
        --predict_file=$DATA_DIR/test.english.jsonlines \
        --do_eval \
        --num_train_epochs=129 \
        --logging_steps=500 \
        --save_steps=3000 \
        --eval_steps=1000 \
        --max_seq_length=4600 \
        --train_file_cache=$DATA_DIR/train.english.4600.pkl \
        --predict_file_cache=$DATA_DIR/test.english.4600.pkl \
        --amp \
        --normalise_loss \
        --max_total_seq_len=5000 \
        --experiment_name=eval_model \
        --warmup_steps=5600 \
        --adam_epsilon=1e-6 \
        --head_learning_rate=3e-4 \
        --learning_rate=1e-5 \
        --adam_beta2=0.98 \
        --weight_decay=0.01 \
        --dropout_prob=0.3 \
        --save_if_best \
        --top_lambda=0.4  \
        --tensorboard_dir=$OUTPUT_DIR/tb \
        --conll_path_for_eval=$DATA_DIR/$SPLIT_FOR_EVAL.english.v4_gold_conll
```

## Training
Train a coreference model using:
```
export OUTPUT_DIR=<output_dir>
export CACHE_DIR=<cache_dir>
export DATA_DIR=<data_dir>

python run_coref.py \
        --output_dir=$OUTPUT_DIR \
        --cache_dir=$CACHE_DIR \
        --model_type=longformer \
        --model_name_or_path=allenai/longformer-large-4096 \
        --tokenizer_name=allenai/longformer-large-4096 \
        --config_name=allenai/longformer-large-4096  \
        --train_file=$DATA_DIR/train.english.jsonlines \
        --predict_file=$DATA_DIR/dev.english.jsonlines \
        --do_train \
        --do_eval \
        --num_train_epochs=129 \
        --logging_steps=500 \
        --save_steps=3000 \
        --eval_steps=1000 \
        --max_seq_length=4096 \
        --train_file_cache=$DATA_DIR/train.english.4600.pkl \
        --predict_file_cache=$DATA_DIR/test.english.4600.pkl \
        --gradient_accumulation_steps=1 \
        --amp \
        --normalise_loss \
        --max_total_seq_len=4600 \
        --experiment_name="s2e_CT-model" \
        --warmup_steps=5600 \
        --adam_epsilon=1e-6 \
        --head_learning_rate=3e-4 \
        --learning_rate=1e-5 \
        --adam_beta2=0.98 \
        --weight_decay=0.01 \
        --dropout_prob=0.3 \
        --save_if_best \
        --top_lambda=0.4  \
        --tensorboard_dir=$OUTPUT_DIR/tb \
	    --t_sim=0.80
        --conll_path_for_eval=$DATA_DIR/dev.english.v4_gold_conll
```

## Cite

If you use this code in your research, please cite our paper:

```
@inproceedings{chai-strube-2022-incorporating,
    title = "Incorporating Centering Theory into Neural Coreference Resolution",
    author = "Chai, Haixia  and
      Strube, Michael",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.218",
    pages = "2996--3002",
    abstract = "In recent years, transformer-based coreference resolution systems have achieved remarkable improvements on the CoNLL dataset. However, how coreference resolvers can benefit from discourse coherence is still an open question. In this paper, we propose to incorporate centering transitions derived from centering theory in the form of a graph into a neural coreference model. Our method improves the performance over the SOTA baselines, especially on pronoun resolution in long documents, formal well-structured text, and clusters with scattered mentions.",
}

```
