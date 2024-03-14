# KC-Based Human-Machine-Code-Translation

## Table of Contents

- [KC-Based-HMCT]()
  - [File directory](#File_directory)
  - [Dataset](#Dataset)
  - [Models](#Models)
  - [Training & Evaluation & Test](#training--evaluation)
  - [Requirement](#Requirement)

## File_directory
```

├─codet5
    ├─AST
    ├─bpe
    ├─codet5_base   //Store the codet5_base pre-training model
    ├─data
    │  └─g4g_functions    //AVATAR-g4g, dataset for testing BISTM
    ├─g4g
    │  ├─java2python
    │  │  ├─biModel-prefix
    │  │  │  └─checkpoint-best-ppl
    │  │  │      └─backward    //Store the backward GTM model
    │  │  ├─bistm
    │  │  │  ├─backward
    │  │  │  │  ├─sample_ast    //Store the generalized backward STM model
    │  │  │  │  └─pytorch_model.bin    //Ungeneralized backward STM model
    │  │  │  └─forward_bao
    │  │  │      ├─sample_ast    //Store the generalized forward STM model
    │  │  │      └─pytorch_model.bin    //Ungeneralized forward STM model
    │  │  ├─paraphrase
    │  │  │  └─checkpoint-best-ppl
    │  │  │      └─GTM    //Store the forward GTM model
    │  │  └─prefix
    │  ├─python2java
    │  │  ├─biModel-prefix
    │  │  │  └─checkpoint-best-ppl
    │  │  │      └─backward    //Store the backward GTM model
    │  │  ├─bistm
    │  │  │  ├─backward
    │  │  │  │  ├─sample_ast    //Store the generalized backward STM model
    │  │  │  │  └─pytorch_model.bin    //Ungeneralized backward STM model
    │  │  │  └─forward_bao
    │  │  │      ├─sample_ast    //Store the generalized forward STM model
    │  │  │      └─pytorch_model.bin    //Ungeneralized forward STM model
    │  │  ├─bistm_data    //Dataset for training BISTM
    │  │  ├─paraphrase
    │  │  │  └─checkpoint-best-ppl
    │  │  │      └─GTM    //Store the forward GTM model
    │  │  └─prefix
    │  │      └─ml_data    //Generate metrics from the result file
    │  └─res_ast    //Relevant experimental results of generalization training
    │      ├─g4g_java-python
    │      ├─g4g_python-java
    │      ├─g4g_sample_java-python
    │      └─g4g_sample_python-java
    ├─models
    │  └─codet5_small
    ├─old_data
    │  └─prefix
    │      └─ml_data
    │          ├─early_stop_525
    │          └─early_stop_g4g_p2j_total
    ├─tensorboard
    │  ├─output
    │  └─test
    │      ├─j2p
    │      │  ├─backward
    │      │  │  └─output
    │      │  └─forward
    │      │      └─output
    │      └─p2j
    │          ├─backward
    │          │  └─output
    │          └─forward
    │              └─output
    ├─test
    │  ├─j2p
    │  │  ├─backward
    │  │  │  ├─cached_data
    │  │  │  └─output
    │  │  ├─forward
    │  │  │  ├─cached_data
    │  │  │  └─output
    │  │  └─result
    │  └─p2j
    │      ├─backward
    │      │  ├─cached_data
    │      │  └─output
    │      ├─forward
    │      │  ├─cached_data
    │      │  └─output
    │      └─result
    ├─transformers
    │  ├─benchmark
    │  │  └─__pycache__
    │  ├─commands
    │  ├─data
    │  │  ├─datasets
    │  │  │  └─__pycache__
    │  │  ├─metrics
    │  │  │  └─__pycache__
    │  │  ├─processors
    │  │  │  └─__pycache__
    │  │  └─__pycache__
    │  └─__pycache__
    └─__pycache__
    
```


## Dataset

We use two datasets to train and test the BiSTM model respectively.

- AVATAR-g4g: The dataset are in the ./data/function directory, mainly for testing.
- Bistm-data: The dataset are in the ./g4g/bistm_data directory, mainly for training.

## Models

#### Models trained from scratch

- GTM model: General machine translation model to generate first-round results.
- BiSTM models: Human-machine collaborative translation model.
  - Forward BiSTM model: Forward decoding.
  - Backward BiSTM model: Backward decoding. 

The BiSTM model is divided into generalized model and ungeneralized model. 

The generalized model takes into account the missing or incomplete validation segment in the case of human feedback.

#### Pre-trained models

- CodeT5_base: CodeT5 pre-training model.
- CodeT5_small: A smaller scale CodeT5 pretrained model.

## Training & Evaluation
Firstly, go to the directory.

```
cd  KC-Based-HMCT/codet5
```

To train, evaluate and test the Bi-STM model, execute the script with the parameters as follows.

```
# To train, evaluate and test the ungeneralized Forward Bi-STM model.

python run_gen_biSTM_ungeneralized.py --do_train --do_eval --do_test --task translate --cache_path 'cache_file_path' --data_dir ./g4g/python2java/bistm_data --res_dir 'result_file_path' --output_dir 'output_file_path' --tokenizer_path ./bpe --model_name_or_path 'pretrained_model'  --save_last_checkpoints  --always_save_model --task translate  --sub_task 'translation_direction' --model_type codet5 --gradient_accumulation_steps 8 --eval_batch_size 1 --max_source_length 510 --max_target_length 510 --beam_size 1 --forward or bcakward

for example:

python run_gen_biSTM_ungeneralized.py --do_train --do_eval --do_test --task translate --cache_path ./test/p2j/forward/cached_data --data_dir ./g4g/python2java/bistm_data --res_dir ./test/p2j/forward/result --output_dir ./test/p2j/forward/output --tokenizer_path ./bpe --model_name_or_path codet5_base  --save_last_checkpoints  --always_save_model --task translate  --sub_task python-java --model_type codet5 --forward  --gradient_accumulation_steps 8 --eval_batch_size 1 --max_source_length 510 --max_target_length 510 --beam_size 1

# To train, evaluate and test the generalized Forward Bi-STM model.

python run_gen_biSTM_generalized.py --do_train --do_eval --do_test --task translate --cache_path 'cache_file_path' --data_dir ./g4g/python2java/bistm_data --res_dir 'result_file_path' --output_dir 'output_file_path' --tokenizer_path ./bpe --model_name_or_path codet5_base  --save_last_checkpoints  --always_save_model --task translate  --sub_task 'translation_direction' --model_type codet5 --gradient_accumulation_steps 8 --eval_batch_size 1 --max_source_length 510 --max_target_length 510 --beam_size 1 --forward or bcakward

for example:

python run_gen_biSTM_generalized.py --do_train --do_eval --do_test --task translate --cache_path ./test/p2j/forward/cached_data --data_dir ./g4g/python2java/bistm_data --res_dir ./test/p2j/forward/result --output_dir ./test/p2j/forward/output --tokenizer_path ./bpe --model_name_or_path codet5_base  --save_last_checkpoints  --always_save_model --task translate  --sub_task python-java --model_type codet5 --forward --beam_size 1 --gradient_accumulation_steps 8 --eval_batch_size 1 --max_source_length 510 --max_target_length 510 --model_name_or_path codet5_base

```

## Requirements

The following packages are version-limited
- 1.5.1 <= torch <= 1.7.0
- transformers==3.0.2
- sacrebleu==1.2.11
- tree_sitter==0.2.1





