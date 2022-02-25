import numpy as np
from data_utils import load_vocab
import constants
import os
import pickle


# NLPLAB_W2V = 'data/w2v_model/wikipedia-pubmed-and-PMC-w2v.bin'
NLPLAB_W2V = 'data/w2v_model/bert_embedding.pkl'
# NLPLAB_W2V = 'data/w2v_model/w2v_retrain.bin'


def export_trimmed_nlplab_vectors(vocab, trimmed_filename, dim=768, bin=NLPLAB_W2V):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
        :param bin:
    """
    # embeddings contains embedding for the pad_tok as well
    embeddings = np.zeros([len(vocab) + 1, dim])
    with open(bin, 'rb') as f:
        emb_dict = pickle.load(f)

        print('bert vocab size', len(emb_dict))

        count = 0
        m_size = len(vocab)
        for word in vocab:
            if word in emb_dict:
                count += 1
                embedding = emb_dict[word]
                word_idx = vocab[word]
                embeddings[word_idx] = embedding
            else:
                pass

    print('Missing rate {}'.format(1.0 * (m_size - count)/m_size))
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


vocab_words = load_vocab(constants.ALL_WORDS)
export_trimmed_nlplab_vectors(vocab_words, 'data/w2v_model/bert_nlplab.npz')

