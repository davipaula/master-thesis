# Text-based Prediction of Popular Click Paths in Wikipedia

This repository contains the code used to run the experiments of the Master thesis `Text-based Prediction of Popular Click Paths in Wikipedia`

## Install (to be updated)

```
# Setup virtual Python 3.6 environment (with conda)
conda create -n smash-rnn python=3.7
conda activate smash-rnn

# Dependencies
pip install -r requirements.txt
```

### Datasets (to be updated)
This project uses a combination of English language Wikipedia articles and Wikipedia English clickstream data. You can find the datasets [here](https://drive.google.com/file/d/1c9ksCCzr6GQnCdZPW0LR4m8TAdCRUbaj). You need to unzip this file in the project root folder.
To download using wget, you can run:

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c9ksCCzr6GQnCdZPW0LR4m8TAdCRUbaj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c9ksCCzr6GQnCdZPW0LR4m8TAdCRUbaj" -O data.zip && rm -rf /tmp/cookies.txt

### Training
- Smash RNN:
To train the model with default parameters, run:

- **python train_smash_rnn.py**

**Options**
- _--num_epochs_: The number of epochs to train the model (default: 1)
- _--batch_size_: The number of articles in each batch (default: 6)
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


You can find trained models in this [link](https://drive.google.com/file/d/1IqkOR7k2t3vjszvCM0SJSPzX6ApijBex/view?usp=sharing). You need to unzip this file in the project root folder.

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IqkOR7k2t3vjszvCM0SJSPzX6ApijBex' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IqkOR7k2t3vjszvCM0SJSPzX6ApijBex" -O trained_models.zip && rm -rf /tmp/cookies.txt

#### Additional models
Additional models were developed to compare the results with Smash-RNN:
- Wikipedia2Vec:
    - Train: **python train_wikipedia2vec.py**
    - Test: **python test_wikipedia2vec.py**
    
- Doc2Vec:
    - Train: **python train_doc2vec.py**
    - Test: **python test_doc2vec.py**
    
#### Results analysis
To generate the tables and figures from Section `Results`, please follow the steps presented in Jupyter Notebook **Results analyzer.ipynb** 