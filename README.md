# ROBT
This is our code repo for paper "Agent4Ranking: Semantic Robust Ranking via
Personalized Query Rewriting Using Multi-agent LLM".
## Files layout
```
    ROBR
    ├── data                    # Dataset files
    │   ├── Industrial          # For industrial dataset in Baidu
    │   └── Trec04              # Robust04 dataset
    ├── Finetuning              # Fituning files
    │   ├── test                # Test files of the fine-tuning models
    │   └── train               # Training files of the fine-tuning models
    ├── model                   # Code impletation of robust model
    │   ├── model_bert.py       # Robust model - bert based backbone model 
    │   ├── model_ernie.py      # Robust model - ernie based backbone model 
    │   ├── model_roberta.py    # Robust model - roberta based backbone model 
    │   ├── test_bert.py        # Robust model test - bert based backbone model 
    │   ├── test_ernie.py       # Robust model test - ernie based backbone model 
    │   └── test_roberta.py     # Robust model test - roberta based backbone model 
    ├── model_path              # model save folder
    │   ├── Pre-trained_model   # pre-trained models    
    │   └── Raw_model           # Raw pre-trained models dowload from Hugging-face  
    └── rewriting               # Query rewrite script
        ├── prompt_cn.py        # For industrial dataset
        └── prompt_en.py        # For public dataset
```
## Usage
TBD
