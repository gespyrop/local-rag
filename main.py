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

    parser.add_argument('question', nargs='?', help='A question to the LLM.')

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
        rag.add(args.add)
        print(f'Added "{args.add}"')

    # Query
    if args.query:
        print('Query:', args.query)

        for result in rag.search(args.query):
            print('\n', 10 * '-', f'{result.source}: {result.id}', 10 * '-')
            print('Chunk:', result.content)

    # Question
    if args.question:
        print('Question:', args.question)

        response = rag.ask(args.question)

        files = set()
        for source in response.sources:
            files.add(source.source)

        print('Answer:', response.content)

        print('\n Sources:')

        for file in files:
            print('-', file)

    # Print help
    if not (args.question or args.add or args.query):
        parser.print_help()
