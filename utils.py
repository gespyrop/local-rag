'''
LocalRAG utils
'''


def chunk_text(text, chunk_size=500, overlap=50) -> list[str]:
    '''
    Break a text into chunks.

    :param text: Input text
    :param chunk_size: Size of a chunk
    :param overlap: Overlap between chunks
    :return: Chunks
    :rtype: list[str]
    '''
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
