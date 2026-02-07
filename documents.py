from pypdf import PdfReader


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


def extract_pdf_text(path: str) -> str:
    '''
    Extrat the text from a PDF file.

    :param path: PDF file path.
    :type path: str
    :return: Extracted text
    :rtype: str
    '''
    reader = PdfReader(path)

    return "\n".join([page.extract_text() for page in reader.pages])
