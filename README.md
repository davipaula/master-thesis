# SMASH-RNN: Semantic Text Matching for Long-Form Documents

A PyTorch implementation of the paper [Semantic Text Match for Long-Form Docuemnts (Jiang et al.)](https://pub-tools-public-publication-data.storage.googleapis.com/pdf/99357ca2ef0d89250e8d0aea47607fc4c556aa09.pdf) is used to predict the number of clicks between Wikipedia articles.

## Install

```
# Setup virtual Python 3.6 environment (with conda)
conda create -n smash-rnn python=3.7
conda activate smash-rnn

# Dependencies
pip install -r requirements.txt
```

### Model

![Model architecure)](smash-rnn_architecture.png)

### Datasets
This project uses a combination of English language Wikipedia articles and Wikipedia English clickstream data. You can find the datasets [here](https://drive.google.com/file/d/1c9ksCCzr6GQnCdZPW0LR4m8TAdCRUbaj). You need to unzip them in the project root folder.

### Training
The model supports three levels:
- word
- sentence
- paragraph (default level)

If you want to train a model with default parameters, you could run:

- **python train_smash_rnn.py --level=word**

If you want to train a model with your preference parameters, like optimizer and learning rate, you could run:

- **python train_smash_rnn.py --num_epoches num_epoches --validation_interval validation_interval**: For example, python train_smash_rnn.py --num_epoches 6 --validation_interval 2

### Test
To test a trained model with your test file, please run the following command:

- **python test_smash_rnn.py --level=word**.

You can find trained models in this [link](https://drive.google.com/open?id=1W62KwmYUSUbRTfeyym6M22IonKRB-i27)

#### Additional models
Additional models were developed to compare the results with Smash-RNN:
- Wikipedia2Vec:
    - Train: **python train_wikipedia2vec.py**
    - Test: **python test_wikipedia2vec.py**
    
- Doc2Vec:
    - Train: **python train_doc2vec.py**
    - Test: **python test_doc2vec.py**
    
#### Results comparison
To compare the results of the models in the evaluation step, please take a look see the Jupyter Notebook **Results analyzer.ipynb** 

### Concepts

The model makes use of the following concepts:

- Word embeddings (Word2Vec)
- Recurrent neural networks (RNN)
- Attention [Paper](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), [Blog post](https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)
- GRU & LSTMs: [Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
- Hierarchical attention network ([Paper](https://www.aclweb.org/anthology/N16-1174), [PyTorch](https://github.com/vietnguyen91/Hierarchical-attention-networks-pytorch/blob/master/src/hierarchical_att_model.py))
- Siamese networks ([PyTorch](https://github.com/MarvinLSJ/LSTM-siamese/))

## Related Links

- [10 free online courses on machine learning](https://twitter.com/chipro/status/1157772112876060672)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html)
- [A Beginner's Guide to LSTMs and Recurrent Neural Networks](https://skymind.ai/wiki/lstm)
