from local_rag.utils import chunk_text


class TestUtils:
    def test_chunk_text(self):
        '''
        Test that text is chunked properly.
        '''
        chunks = chunk_text("ABCDEFGHI", chunk_size=3, overlap=1)

        assert chunks == ['ABC', 'CDE', 'EFG', 'GHI', 'I']
