from pathlib import Path
from typing import Dict, List, Any
from xml.etree.ElementTree import ParseError

import pytest

from target_extraction.dataset_parsers import semeval_2016
from target_extraction.data_types_util import Span

class TestSemeval2016:

    DATA_PATH_DIR = Path(__file__, '..', '..', 'data', 'parsing', 'semeval_16').resolve()

    def _target_answer(self, conflict: bool) -> Dict[str, Dict[str, Any]]:
        answer_0 = {'text_id': '1004293:0', 'text': "The staff were ok", 
                    'targets': ['staff'], 'spans': [Span(4, 9)],
                    'target_sentiments': ['neutral'],
                    'categories': ['service']}
        answer_1 = {'text_id': '1004293:1', 'text': "The day was ok but the food was great", 
                    'targets': [None, 'food'], 'spans': [Span(0, 0), Span(23, 27)],
                    'target_sentiments': ['positive', 'negative'], 
                    'categories': ['anecdotes/miscellaneous', 'food']}
        answer_2 = {'text_id': '1016296:0', 'text': "I had a horrible waiter today food",
                    'targets': ['waiter', 'food'], 
                    'spans': [Span(17, 23), Span(30, 34)],
                    'target_sentiments': ['negative', 'conflict'],
                    'categories': ['service', 'food']}
        answer_3 = {'text_id': '1016296:1', 'text': "Nothing really happened today",
                    'targets': None, 'spans': None, 'target_sentiments': None}
        answer_4 = {'text_id': '1016296:2', 'text': "The drinks were neither good or bad bu the check in was awful", 
                    'targets': ['drinks', 'check'], 'spans': [Span(4, 10), Span(43, 48)],
                    'target_sentiments': ['conflict', 'positive'],
                    'categories': ['drinks', 'service']}
        if conflict:
            answer_4['target_sentiments'] = ['positive']
            answer_4['spans'] = [Span(43, 48)]
            answer_4['targets'] = ['check']
            answer_4['category_sentiments'] = None
            answer_4['categories'] = ['service']

            answer_2['target_sentiments'] = ['negative']
            answer_2['targets'] = ['waiter']
            answer_2['spans'] = [Span(17, 23)]
            answer_2['categories'] = ['service']
            answer_2['category_sentiments'] = None
        return {'3121': answer_0, '2777': answer_1, '2534': answer_2,
                '2634': answer_3, '1793': answer_4}
        

    @pytest.mark.parametrize("conflict", (True, False))
    def test_read_from_file(self, conflict: bool):
        data_fp = Path(self.DATA_PATH_DIR, 'semeval_16_example.xml')
        target_text_collection = semeval_2016(data_fp, conflict)

        assert len(target_text_collection) == 5

        _ids = ['3121', '2777', '2534', '2634', '1793']
        assert list(target_text_collection.keys()) == _ids
        true_answers = self._target_answer(conflict=conflict)
        for answer_key, answer in true_answers.items():
            test_answer = target_text_collection[answer_key]
            for key in answer:
                assert answer[key] == test_answer[key]

    @pytest.mark.parametrize("conflict", (True, False))
    def test_unreadable_file(self, conflict: bool):
        # Test that it will raise a ParseError is not formatted correctly e.g. 
        # Contains mismatched tags

        unreadable_fp = Path(self.DATA_PATH_DIR, 
                             'unpassable_semeval_16_example.xml')
        with pytest.raises(ParseError):
            semeval_2016(unreadable_fp, conflict)

    @pytest.mark.parametrize("conflict", (True, False))
    def test_not_semeval_file(self, conflict: bool):
        # Test that it will raise a SyntaxError as the file does not follow 
        # SemEval format.

        unreadable_fp = Path(self.DATA_PATH_DIR, 
                             'not_semeval_16_example.xml')
        with pytest.raises(SyntaxError):
            semeval_2016(unreadable_fp, conflict)

    