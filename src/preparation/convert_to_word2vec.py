from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


def convert(
    glove_file_path: str,
    output_file_path: str,
) -> None:
    """Converts GloVe embeddings to Word2Vec txt format

    Parameters
    ----------
    glove_file_path : str
        Path of GloVe file

    output_file_path :
        Path to save Word2Vec txt file
    """
    _glove_file = datapath(glove_file_path)
    _output_file = get_tmpfile(output_file_path)
    _ = glove2word2vec(_glove_file, _output_file)
