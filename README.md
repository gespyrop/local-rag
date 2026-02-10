# Local Retrieval-Augmented Generation (RAG)
Create a local ***Retrieval-Augmented Generation (RAG)*** system through a single configuration file.

## Configuration
```yaml
# Chunk properties
chunk:
  size: 500
  overlap: 100

# Vector database properties
vector:
  db: chroma
  collection: documents
```

## Usage
### Create a `RAGService`
```python
from local_rag import RAGService

rag = RAGService.from_yaml_config('config.yaml')
```

### Add documents
#### Add a document
```python
rag.add('docs/lorem_ipsum.pdf')
```

#### Add all documents in a directory recursively
```python
rag.add('docs/')
```

### Search the vector database
```python
rag.search('typesetting industry')
```

## Embedding model
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Supported vector databases
- [Chroma](https://docs.trychroma.com/docs/overview/getting-started)
