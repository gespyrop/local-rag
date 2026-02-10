'''
High-level RAG logic.
'''
import os
from typing import Self
from omegaconf import OmegaConf
from .vector import VectorDatabase, VectorQueryResult, vector_database_factory
from .embedding import get_text_embeddings
from .parsers import parser_factory
from .utils import chunk_text


class RAGService:
    '''
    Implements Retrieval-Augmented Generation steps by encapsulating
    and orchestrating the required set of services.
    '''

    def __init__(self, vector_db: VectorDatabase,
                 chunk_size: int, chunk_overlap: int):
        self.vector_db = vector_db
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def from_yaml_config(yaml_file: str) -> Self:
        '''
        Docstring for from_yaml_config

        :param yaml_file: Description
        :type yaml_file: str
        :return: Description
        :rtype: Any
        '''
        config = OmegaConf.load(yaml_file)

        # Chunk configuration
        chunk_config = config.get('chunk', {})
        chunk_size = chunk_config.get('size', 500)
        chunk_overlap = chunk_config.get('overlap', 100)

        # Vector database configuration
        vector_config = config.get('vector', {})
        db_name = vector_config.get('db', 'chroma')
        vector_db = vector_database_factory(db_name, **vector_config)

        return RAGService(vector_db, chunk_size, chunk_overlap)

    def add(self, path: str):
        '''
        Add a document or a directory of documents to the vector database.

        :param path: Path to be added
        :type path: str
        '''
        if os.path.isdir(path):
            self.add_directory(path)
        else:
            self.add_document(path)

    def add_directory(self, directory: str):
        '''
        Recursively add all documents in a directory to the vector database.

        :param directory: Directory to be added
        :type directory: str
        '''
        for document in os.listdir(directory):
            document_path = os.path.join(directory, document)
            self.add_document(document_path)

    def add_document(self, document: str):
        '''
        Add a document to the vector database.

        :param document: Document file path
        :type document: str
        '''
        file_name, file_extension = document.split('/')[-1].split('.')
        file_parser = parser_factory(file_extension)
        text = file_parser(document)

        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)

        ids = [f'{file_name}_chunk_{chunk+1}' for chunk in range(len(chunks))]
        metadatas = [{"source": document} for _ in chunks]
        embeddings = get_text_embeddings(chunks)

        self.vector_db.add(ids, embeddings, metadatas, documents=chunks)

    def search(self, query: str, k: int = 3) -> list[VectorQueryResult]:
        '''
        Search for the `k` most relevant chunks in the database.

        :param query: Search query
        :type query: str
        :param k: Number of results
        :type k: int
        :return: List containing the `k` most relevant results.
        :rtype: list[VectorQueryResult]
        '''
        embedding = get_text_embeddings(query)

        return self.vector_db.query(embedding, k)
