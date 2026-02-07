from documents import chunk_text, extract_pdf_text


class TestDocuments:
    def test_chunk_text(self):
        chunks = chunk_text("ABCDEFGHI", chunk_size=3, overlap=1)

        assert chunks == ['ABC', 'CDE', 'EFG', 'GHI', 'I']

    def test_extract_pdf(self):
        assert extract_pdf_text('test/test.pdf') == 'test'
