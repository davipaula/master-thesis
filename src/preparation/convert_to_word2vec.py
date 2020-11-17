from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


def convert(
    glove_file_path="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/glove.6B.200d.txt",
    output_file_path="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/glove.6B.200d.w2vformat.txt",
):
    _glove_file = datapath(glove_file_path)
    _output_file = get_tmpfile(output_file_path)
    _ = glove2word2vec(_glove_file, _output_file)


if __name__ == "__main__":
    convert()
