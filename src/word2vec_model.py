# Import required libraries
import pandas as pd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from gensim.models.doc2vec import LabeledSentence


def extract_unique_documents(dataset):
    current_texts = dataset[['current_article', 'current_article_text']]
    previous_texts = dataset[['previous_article', 'previous_article_text']]

    current_texts.columns = ['title', 'text']
    previous_texts.columns = ['title', 'text']

    combined_texts = current_texts.append(previous_texts, sort=False, ignore_index=True)

    unique_documents = combined_texts.drop_duplicates().reset_index(drop=True)

    return unique_documents


def process_questions(questions):
    question_list = [clean_text(question, False) for question in questions]

    return question_list


def clean_text(text, should_remove_stopwords):
    text = text.lower().split()

    if should_remove_stopwords:
        text = remove_stopwords(text)

    text = " ".join(text)

    text = remove_special_characters(text)

    return text


def remove_stopwords(text):
    stops = set(stopwords.words("english"))
    words = [w for w in text.lower().split() if not w in stops]

    return " ".join(words)


def remove_special_characters(text):
    text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", text)
    text = re.sub(r"\'s", " 's ", text)
    text = re.sub(r"\'ve", " 've ", text)
    text = re.sub(r"n\'t", " 't ", text)
    text = re.sub(r"\'re", " 're ", text)
    text = re.sub(r"\'d", " 'd ", text)
    text = re.sub(r"\'ll", " 'll ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


def generate_tagged_documents(documents):
    tagged_documents = []

    for i in range(len(documents)):
        tagged_documents.append(TaggedDocument(documents['text'][i].split(), [i]))

    return tagged_documents


def generate_split_documents(documents):
    documents.columns = ['title', 'text']
    documents['text'] = [document.split() for document in documents['text']]

    return documents


def generate_cosine_similarity_scores(current_article_text, previous_article_text):
    return [model.wv.n_similarity(current_article_text[i], previous_article_text[i]) for i in
            range(len(current_article_text))]


if __name__ == '__main__':
    model = Doc2Vec(dm=1, min_count=1, window=10, vector_size=150, sample=1e-4, negative=10)

    df = pd.read_csv('../data/wiki_df_text.csv', nrows=50)

    # df = df.drop(df[df.isnull().any(axis=1)].index, inplace=True)

    unique_documents = extract_unique_documents(df)
    labeled_questions = generate_tagged_documents(unique_documents)
    model.build_vocab(labeled_questions)

    current_article_split = generate_split_documents(df[['current_article', 'current_article_text']])
    previous_article_split = generate_split_documents(df[['previous_article', 'previous_article_text']])

    scores = generate_cosine_similarity_scores(current_article_split['text'], previous_article_split['text'])

    print(scores)

    # # Train the model with 20 epochs
    #
    # for epoch in range(20):
    #     model.train(labeled_questions, epochs=model.iter, total_examples=model.corpus_count)
    #     print("Epoch #{} is complete.".format(epoch + 1))
    #
    # model.most_similar('washington')
    #
    # score = model.n_similarity(questions1_split[i], questions2_split[i])
    #
    # from sklearn.metrics import accuracy_score
    #
    # accuracy = accuracy_score(df.is_duplicate, scores) * 100
