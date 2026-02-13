'''
Vector embeddings.
'''
from sentence_transformers import SentenceTransformer
from transformers import logging

logging.set_verbosity_error()


class EmbeddingService:
    '''
    Embedding service.
    '''
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self,
              sentences: str | list[str]) -> list[float] | list[list[float]]:
        '''
        Get sentence embeddings using a sentence transformer.

        :param sentences: Input sentences
        :type sentences: str | list[str]
        :return: Sentence embeddings
        :rtype: list[float] | list[list[float]]
        '''
        embeddings = self.model.encode(sentences)
        return embeddings.tolist()
