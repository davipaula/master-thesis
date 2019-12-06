import gensim

import torch
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

# config variables (can be changed)
max_words_per_sentence = 32
max_sentences_per_paragraph = 16
max_paragraphs_per_doc = 12
batch_size = 6

# generate random data (word ids from 1-999)
word_ids = torch.randint(1, 999, (batch_size,max_paragraphs_per_doc,max_sentences_per_paragraph,max_words_per_sentence))

# remember the length of each sequence
words_per_sentence = torch.randint(1, max_words_per_sentence, (batch_size,max_paragraphs_per_doc,max_sentences_per_paragraph))
sentences_per_paragraph = torch.randint(1, max_sentences_per_paragraph, (batch_size,max_paragraphs_per_doc))
paragraphs_per_doc = torch.randint(1, max_paragraphs_per_doc, (batch_size,))

# data loader should return something like this
batch_data = (word_ids, words_per_sentence, sentences_per_paragraph, paragraphs_per_doc)
# ...

# the model class (implement this as nn.module)
# __init__()
# - hidden sizes can be changed

# Load from txt file (in word2vec format)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./data/glove.6B.200d.w2vformat.1k.txt')

# Convert to PyTorch tensor
weights = torch.FloatTensor(w2v_model.vectors)

# Init embedding layer
embedding = nn.Embedding.from_pretrained(weights)
embed_dim = w2v_model.vector_size

# RNN + attention layers
word_gru_hidden_size = 100
word_gru_out_size = word_gru_hidden_size * 2
word_gru = nn.GRU(embed_dim, word_gru_hidden_size, bidirectional=True, batch_first=True)
word_attention = nn.Linear(word_gru_out_size, 50)
word_context_vector = nn.Linear(50, 1, bias=False)  # Word context vector to take dot-product with

sentence_gru_hidden_size = 200
sentence_gru_out_size = sentence_gru_hidden_size * 2
sentence_gru = nn.GRU(word_gru_out_size, sentence_gru_hidden_size, bidirectional=True, batch_first=True)
sentence_attention = nn.Linear(sentence_gru_out_size, 50)
sentence_context_vector = nn.Linear(50, 1, bias=False)

paragraph_gru_hidden_size = 300
paragraph_gru_out_size = paragraph_gru_hidden_size * 2
paragraph_gru = nn.GRU(sentence_gru_out_size, paragraph_gru_hidden_size, bidirectional=True, batch_first=True)
paragraph_attention = nn.Linear(paragraph_gru_out_size, 50)
paragraph_context_vector = nn.Linear(50, 1, bias=False)

# --> forward(word_ids, words_per_sentence, sentences_per_paragraph, paragraphs_per_doc)

# actual size of input
_batch_size = word_ids.size()[0]
_paragraphs_per_doc = word_ids.size()[1]
_sentences_per_paragraph = word_ids.size()[2]

# zero placeholders
sentences = torch.zeros((_batch_size, _paragraphs_per_doc, _sentences_per_paragraph, word_gru_out_size))  # TODO device=self.get_single_device() for GPU support
paragraphs = torch.zeros((_batch_size, _paragraphs_per_doc, sentence_gru_out_size))
docs = torch.zeros((_batch_size, paragraph_gru_out_size))

# iterate over each hierarchy level
for paragraph_idx in range(_paragraphs_per_doc):
    for sentence_idx in range(_sentences_per_paragraph):
        # attention over words
        word_ids_in_sent = word_ids[:, paragraph_idx, sentence_idx, :]  # 1st dim = batch, last dim = words
        words_in_sent = embedding(word_ids_in_sent)  # get word embeddings from ids

        # pack padded sequence
        packed_words = pack_padded_sequence(words_in_sent, lengths=words_per_sentence[:,paragraph_idx,sentence_idx].tolist(), batch_first=True, enforce_sorted=False)

        word_gru_out, _ = word_gru(packed_words)

        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        word_att_out = torch.tanh(word_attention(word_gru_out.data))

        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        word_att_out = word_context_vector(word_att_out).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = word_att_out.max()  # scalar, for numerical stability during exponent calculation
        word_att_out = torch.exp(word_att_out - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        word_att_out, _ = pad_packed_sequence(PackedSequence(data=word_att_out,
                                                               batch_sizes=word_gru_out.batch_sizes,
                                                               sorted_indices=word_gru_out.sorted_indices,
                                                               unsorted_indices=word_gru_out.unsorted_indices),
                                                batch_first=True)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = word_att_out / torch.sum(word_att_out, dim=1,
                                                 keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
        _sentence, _ = pad_packed_sequence(word_gru_out,
                                           batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        # gets the representation for the sentence
        _sentence = (_sentence.float() * word_alphas.unsqueeze(2)).sum(dim=1)  # (batch_size, word_gru_out_size)

        sentences[:,paragraph_idx,sentence_idx] = _sentence

    # attention over sentences
    sentences_in_paragraph = sentences[:,paragraph_idx,:]

    # pack padded sequence of sentences
    packed_sentences = pack_padded_sequence(sentences_in_paragraph,
                                        lengths=sentences_per_paragraph[:, paragraph_idx].tolist(),
                                        batch_first=True, enforce_sorted=False)

    sentence_gru_out, _ = sentence_gru(packed_sentences)

    sent_att_out = torch.tanh(sentence_attention(sentence_gru_out.data))

    # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
    sent_att_out = sentence_context_vector(sent_att_out).squeeze(1)  # (n_words)

    # Compute softmax over the dot-product manually
    # Manually because they have to be computed only over words in the same sentence

    # First, take the exponent
    max_value = sent_att_out.max()  # scalar, for numerical stability during exponent calculation
    sent_att_out = torch.exp(sent_att_out - max_value)  # (n_words)

    # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
    sent_att_out, _ = pad_packed_sequence(PackedSequence(data=sent_att_out,
                                                         batch_sizes=sentence_gru_out.batch_sizes,
                                                         sorted_indices=sentence_gru_out.sorted_indices,
                                                         unsorted_indices=sentence_gru_out.unsorted_indices),
                                          batch_first=True)  # (n_sentences, max(words_per_sentence))

    # Calculate softmax values as now words are arranged in their respective sentences
    sent_alphas = sent_att_out / torch.sum(sent_att_out, dim=1,
                                           keepdim=True)  # (n_sentences, max(words_per_sentence))

    # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
    _paragraph, _ = pad_packed_sequence(sentence_gru_out,
                                       batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

    # Find sentence embeddings
    # gets the representation for the sentence
    _paragraph = (_paragraph.float() * sent_alphas.unsqueeze(2)).sum(dim=1)  # (batch_size, word_gru_out_size)

    paragraphs[:, paragraph_idx] = _paragraph

# attention over paragraphs
# paragraphs

# pack padded sequence of sentences
packed_paragraphs = pack_padded_sequence(paragraphs,
                                    lengths=paragraphs_per_doc.tolist(),
                                    batch_first=True, enforce_sorted=False)

paragraph_gru_out, _ = paragraph_gru(packed_paragraphs)

# This implementation uses the feature sentence_embeddings. Paper uses hidden state
paragraph_att_out = torch.tanh(paragraph_attention(paragraph_gru_out.data))

# Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
paragraph_att_out = paragraph_context_vector(paragraph_att_out).squeeze(1)  # (n_words)

# Compute softmax over the dot-product manually
# Manually because they have to be computed only over words in the same sentence

# First, take the exponent
max_value = paragraph_att_out.max()  # scalar, for numerical stability during exponent calculation
paragraph_att_out = torch.exp(paragraph_att_out - max_value)  # (n_words)

# Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
paragraph_att_out, _ = pad_packed_sequence(PackedSequence(data=paragraph_att_out,
                                                     batch_sizes=paragraph_gru_out.batch_sizes,
                                                     sorted_indices=paragraph_gru_out.sorted_indices,
                                                     unsorted_indices=paragraph_gru_out.unsorted_indices),
                                      batch_first=True)  # (n_sentences, max(words_per_sentence))

# Calculate softmax values as now words are arranged in their respective sentences
paragraph_alphas = paragraph_att_out / torch.sum(paragraph_att_out, dim=1,
                                       keepdim=True)  # (n_sentences, max(words_per_sentence))

# Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
doc, _ = pad_packed_sequence(paragraph_gru_out,
                                   batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

# Find document embeddings
doc = (doc.float() * paragraph_alphas.unsqueeze(2)).sum(dim=1)  # (batch_size, paragraph_gru_out_size)

## doc is what your `get_document_representation` method should return

