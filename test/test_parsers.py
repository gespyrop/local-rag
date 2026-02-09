from local_rag.parsers import parser_factory


class TestParsers:
    def test_parse_pdf(self):
        '''
        Test that the test from a PDF file is extracted correctly
        '''
        parser = parser_factory('pdf')
        assert parser('test/test_files/test.pdf') == 'test'

    def test_parse_txt(self):
        '''
        Test that the test from a .txt file is extracted correctly
        '''
        parser = parser_factory('txt')
        assert parser('test/test_files/test.txt') == 'test'
