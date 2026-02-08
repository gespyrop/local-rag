'''
Vector database integrations.
'''
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, Mapping

import chromadb


@dataclass
class VectorQueryResult:
    '''
    Represents a single query result
    '''
    id: str
    source: str
    content: str
    distance: float


class VectorDatabase(Protocol):
    '''
    Abstract Vector Database
    '''

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
        documents: Sequence[str],
    ) -> None:
        '''
        Add documents to the database.

        :param ids: Document identifiers
        :type ids: Sequence[str]
        :param embeddings: Document embeddings
        :type embeddings: Sequence[Sequence[float]]
        :param documents: Document contents
        :type documents: Sequence[str]
        :param metadatas: Document metadata
        :type metadatas: Sequence[Mapping[str, Any]]
        '''
        ...

    def query(self,
              embedding: Sequence[float],
              k: int = 3) -> list[VectorQueryResult]:
        '''
        Query documents from the database.

        :param embedding: Query embedding
        :type embedding: Sequence[float]
        :param k: Number of results
        :type k: int
        :return: k most similar documents
        :rtype: list[VectorQueryResult]
        '''
        ...


# Maps keys to VectorDatabase subclasses
registry: dict[str, VectorDatabase] = {}


def register(key: str):
    '''
    Decorator that registers a VectorDatabase subclass under a given key.

    :param key: A unique key for VectorDatabase subclasses
    :type key: str
    '''
    def decorator(cls):
        registry[key] = cls

    return decorator


def vector_database_factory(key: str, *args, **kwargs) -> VectorDatabase:
    '''
    Factory function that returns the `VectorDatabase` instance
    registered under the provided key.

    :param key: Key under which the `VectorDatabase` instance is registered.
    (e.g. `'chroma'`)
    :type key: str
    :return: `VectorDatabase` instance registered under the provided `key`.
    :rtype: VectorDatabase
    '''
    if key not in registry:
        raise KeyError(f'"{key}" is not registered as a vector database')

    return registry[key](*args, **kwargs)


@register('chroma')
class ChromaDatabase(VectorDatabase):
    '''
    Chroma Vector Database
    '''

    def __init__(self, **kwargs):
        collection_name = kwargs.get('collection', 'documents')
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids, embeddings, metadatas, documents):
        self.collection.add(ids, embeddings, metadatas, documents)

    def query(self, embedding, k=3):
        # Query result
        qr = self.collection.query(query_embeddings=[embedding], n_results=k)

        # Number of returned results
        num_results = len(qr['ids'][0])

        # Final results
        results: list[VectorQueryResult] = []

        for index in range(num_results):
            results.append(
                VectorQueryResult(
                    qr['ids'][0][index],
                    qr['metadatas'][0][index]['source'],
                    qr['documents'][0][index],
                    qr['distances'][0][index]
                )
            )

        return results
