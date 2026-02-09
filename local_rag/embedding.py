'''
Vector embedding functions.
'''
from sentence_transformers import SentenceTransformer
from transformers import logging

logging.set_verbosity_error()


def get_text_embeddings(
        sentences: str | list[str]) -> list[float] | list[list[float]]:
    '''
    Get sentence embeddings using a sentence transformer.

    :param sentences: Input sentences
    :type sentences: str | list[str]
    :return: Sentence embeddings
    :rtype: list[float] | list[list[float]]
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings.tolist()
