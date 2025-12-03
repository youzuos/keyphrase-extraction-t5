# AI-Powered Keyword Extraction System

A keyword extraction system based on LLM, trained and evaluated using the KP20K dataset.

## Structure

```
Doc_1/
├── data/kp20k/          # dataset
├── src/                 
│   ├── data/            # data processing
│   ├── models/          # model training code
│   └── api/             # API
├── requirements.txt     # libraries
├── static/              # frontend
├── models/              # download from Hugging Face (youzuos/Keyword_Extraction_T5)
│   └── final_model/     # trained model
└── README.md           # document

stall library
pip install -r requirements.txt

1. data processing
The data processing module can be used to load, preprocess, and analyze the KP20K dataset.


2. model training
python train.py
Configuration can be adjusted in `src/models/config.py` ：
- model size（t5-small, t5-base, t5-large）
- training epochs
- batch size
- learning rate

3. use trained model
interactive prediction
python predict.py --model_path models/checkpoints/final_model --interactive


Single text prediction
python predict.py --model_path models/checkpoints/final_model --text "Your text here"


4. evaluation
python evaluate.py --model_path models/checkpoints/final_model --sample_size 100


5. API Service
python api.py

- API documents：http://localhost:8000/docs
- health：http://localhost:8000/health
- extract keywords：POST http://localhost:8000/extract

6. Web
python api.py

Open in browser：
http://localhost:8000/

The front-end interface provides simple text input and keyword display functions.

dataset format
KP20K dataset contains the titles, abstracts, and keywords of scientific papers. Each line is in JSON format:
- id
- title`
- abstract（Used）
- keyphrases（Used）
- prmu






