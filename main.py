'''
Local RAG
'''
import argparse

from vector import VectorQueryResult, vector_database_factory
from embedding import get_text_embeddings
from parsers import parser_factory
from utils import chunk_text


CHUNK_SIZE = 50
CHUNK_OVERLAP = 10


def add_file(file: str):
    '''
    Add a file to the vector database.

    :param file: File path
    :type file: str
    '''
    file_name, file_extension = file.split('/')[-1].split('.')
    file_parser = parser_factory(file_extension)
    text = file_parser(file)

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    ids = [f'{file_name}_chunk_{chunk+1}' for chunk in range(len(chunks))]
    metadatas = [{"source": file} for _ in chunks]
    embeddings = get_text_embeddings(chunks)

    db = vector_database_factory('chroma', collection='documents')
    db.add(ids, embeddings, metadatas, documents=chunks)


def search(query: str, k: int = 3) -> list[VectorQueryResult]:
    '''
    Search for the `k` most relevant chunks in the database.

    :param query: Search query
    :type query: str
    :param k: Number of results
    :type k: int
    :return: Dictionary containing the `k` most relevant chunks.
    :rtype: dict[str, Any]
    '''
    embedding = get_text_embeddings(query)

    db = vector_database_factory('chroma', collection='documents')
    return db.query(embedding, k)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='LocalRAG',
        description='Local retrieval augmented generation'
    )

    parser.add_argument(
        '-a', '--add', help='Add a new file to the vector database.')
    parser.add_argument('-q', '--query', help='Query the vector database.')

    args = parser.parse_args()

    # Add new document
    if args.add:
        if not args.add.endswith('.pdf'):
            print("Unsupported file type. Only PDF files are supported.")
            exit(1)

        # Add document
        add_file(args.add)

        print(f'Added "{args.add}"')

    # Query
    elif args.query:
        print('Query:', args.query)

        for result in search(args.query):
            print('\n', 10 * '-', f'{result.source}: {result.id}', 10 * '-')
            print('Chunk:', result.content)

    # Print usage message
    else:
        parser.print_help()
