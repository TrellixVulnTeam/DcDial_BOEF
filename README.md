# KnowledGPT

This is an implementation of our paper
- Semantic-Oriented Knowledge-Aware Document-Grounded Conversation with GPT-2

## Requirements

- Python 3.6
- Pytorch 1.2.0
- CUDA 10.0 supported GPU with at least 12GB memory
- see [requirements.txt](requirements.txt) for more details

## Usage

To run our pretrained model on CMU_DoG:

- Download the preprocessed data from [here](https://drive.google.com/file/d/16SbW7fEiAjofMijMcDLiMgoeJxFTlMGY/view?usp=sharing).
- Download the checkpoint from [here](https://drive.google.com/drive/folders/1N7nG0fZqd3eMr_zKt025yHNCk8l9jH4w?usp=sharing), then save to "runs/dcdialog"

- To evaluate the model, run
```bash
python dcdialog_movie.py
```


