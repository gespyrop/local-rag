from parsers import parser_factory


class TestParsers:
    def test_parse_pdf(self):
        '''
        Test that the test from a PDF file is extracted correctly
        '''
        parser = parser_factory('pdf')
        assert parser('test/test.pdf') == 'test'
