'''
Local RAG
'''
from embedding import get_text_embeddings


if __name__ == "__main__":
    embedding = get_text_embeddings(["Hello from localrag!"])
    print(embedding)
