'''
Local RAG
'''
from documents import extract_pdf_text, chunk_text
from embedding import get_text_embeddings


if __name__ == "__main__":
    text = extract_pdf_text('test/test.pdf')
    chunks = chunk_text(text, 2, 1)

    embedding = get_text_embeddings(chunks)
    print(len(embedding))
    print([len(e) for e in embedding])
