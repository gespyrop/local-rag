'''
Local RAG
'''
import argparse
from local_rag import RAGService

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog='LocalRAG',
        description='Local Retrieval-Augmented Generation.'
    )

    parser.add_argument(
        '-a', '--add',
        help='Add a new file to the vector database.'
    )

    parser.add_argument(
        '-q', '--query',
        help='Query the vector database.'
    )

    parser.add_argument(
        '-c', '--config',
        help='YAML configuration file path.',
        default='config.yaml'
    )

    args = parser.parse_args()

    # Create RAG service
    rag: RAGService = RAGService.from_yaml_config(args.config)

    # Add new document
    if args.add:
        rag.add_document(args.add)
        print(f'Added "{args.add}"')

    # Query
    elif args.query:
        print('Query:', args.query)

        for result in rag.search(args.query):
            print('\n', 10 * '-', f'{result.source}: {result.id}', 10 * '-')
            print('Chunk:', result.content)

    # Print usage message
    else:
        parser.print_help()
