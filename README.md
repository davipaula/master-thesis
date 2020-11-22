# Text-based Prediction of Popular Click Paths in Wikipedia

This repository contains the code used to run the experiments of the Master thesis `Text-based Prediction of Popular Click Paths in Wikipedia`

## Install

```
# Setup virtual Python 3.7 environment (with conda)
conda create -n smash-rnn python=3.7
conda activate smash-rnn

# Install dependencies
pip install -r requirements.txt
```

### Datasets
This project uses a combination of English language Wikipedia articles and Wikipedia English clickstream data. 

#### Using prepared datasets:

- Download the datasets used in this experiment [here](https://github.com/malteos/thesis-davi/releases/download/0.2-alpha/data.zip). You need to unzip this file in the project root folder.

- Unzip the file in the root folder of the project    
    
#### Creating datasets from scratch

To create the datasets from scratch, you need to run the steps below. Please note that:
- This process may take up to 8 hours to complete when running on a server (data extraction and tokenization are the most time consuming tasks).
- The availability of Wikipedia dumps is limited - in general only the last 3-5 dumps are available.
- This process was tested only with the English language, and Simple English Wikipedia dumps. 

1. Download Clickstream data: https://dumps.wikimedia.org/other/clickstream/
1. Download Wikipedia dump: https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/
1. Move the Clickstream and Wikipedia dump files into the folder `./data/source`
1. In the root folder, run the script `dataset_creator.py`
```bash
python dataset_creator.py
``` 

### Using models
You can find trained models in this [link](https://github.com/malteos/thesis-davi/releases/download/0.2-alpha/trained_models.zip). You need to unzip this file in the project root folder.


#### Training from scratch
- Smash RNN:
To train the model with default parameters, go to the root of the application and run:

```
python train_smash_rnn.py
```

**Options**
- _--num_epochs_: The number of epochs to train the model (default: 1)
- _--batch_size_: The number of articles in each batch. This value needs to be small when using the complete article structure due to memory limitation issues  (default: 6)
- _--level_: The deepest level of Smash RNN. Possible choices are `word`, `sentence` or `paragraph` (default: `paragraph`)
- _--paragraphs_limit_: Maximum number of paragraphs per article that will be processed (default: 300)
- _--model_name_: Name to use when saving the model and results (default: `base`)
- _--w2v_dimension_: Number of dimensions of Word2Vec (default: 50)
- _--introduction_only_: Whether the model should use only the introduction section or the complete text of the article (default: `False`)

### Test
To test the model with default parameters, run:

- **python test_smash_rnn.py --model_name=NAME_OF_THE_MODEL_TRAINED --level=word**.

**Options**
- _--batch_size_: The number of articles in each batch (default: 6)
- _--level_: The deepest level of Smash RNN. Possible choices are `word`, `sentence` or `paragraph` (default: `paragraph`)
- _--paragraphs_limit_: Maximum number of paragraphs per article that will be processed (default: 300)
- _--model_name_: Name to use when saving the model and results. Should match the name of a trained model (default: `base`)
- _--w2v_dimension_: Number of dimensions of Word2Vec (default: 50)
- _--introduction_only_: Whether the model should use only the introduction section or the complete text of the article (default: `False`)


#### Additional models
Additional models were developed to compare the results with Smash-RNN:
- Wikipedia2Vec:
    1. First you need to learn the embeddings for the Wikipedia2Vec model. See instructions in https://wikipedia2vec.github.io/wikipedia2vec/commands/
     
    - Train: 
    
    ```python train_wikipedia2vec.py```
    
    - Test: 
    
    ```python test_wikipedia2vec.py```
    
- Doc2Vec:
    - Train: 
    
    ```python train_doc2vec.py```
    
    - Test: 
    
    ```python test_doc2vec.py```
    
#### Results analysis
To generate the tables and figures from Section `Results` of the thesis, please follow the steps presented in Jupyter Notebook ``Results analyzer.ipynb`` 