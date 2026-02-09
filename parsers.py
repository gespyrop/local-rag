'''
Parsers for various file extensions.
'''
from typing import Callable
from pypdf import PdfReader

# Maps keys to parser callables
registry: dict[str, Callable] = {}


def register(extension: str):
    '''
    Decorator that registers a parser callable
    (e.g. function or class implementing the `__call__` method)
    for a given file extension.

    :param extension: The file extension for which the parser
    will be registered.
    :type extension: str
    '''
    def decorator(parser):
        registry[extension] = parser

    return decorator


def parser_factory(extension: str) -> Callable:
    '''
    Factory function that returns a parser for the given file extension.

    :param extension: File extension.
    :type extension: str
    :return: Parser for the given file extension.
    :rtype: Callable
    '''
    if extension not in registry:
        raise KeyError(f'No registered parsers found for "{extension}" files.')

    parser = registry[extension]

    # Check for class-based parser or function-based parser
    return parser() if isinstance(parser, type) else parser


@register('pdf')
def parse_pdf(path: str) -> str:
    '''
    Extract the text from a PDF file.

    :param path: PDF file path.
    :type path: str
    :return: Extracted text
    :rtype: str
    '''
    reader = PdfReader(path)

    return "\n".join([page.extract_text() for page in reader.pages])


@register('txt')
def parse_txt(path: str) -> str:
    '''
    Extract the text from a txt file.

    :param path: File path.
    :type path: str
    :return: Extracted text
    :rtype: str
    '''
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
