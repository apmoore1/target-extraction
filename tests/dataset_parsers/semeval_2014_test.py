from pathlib import Path
from typing import Dict, List, Any
from xml.etree.ElementTree import ParseError

import pytest

from target_extraction.dataset_parsers import semeval_2014
from target_extraction.data_types_util import Span

class TestSemeval2014:

    DATA_PATH_DIR = Path(__file__, '..', '..', 'data', 'parsing', 'semeval_14').resolve()

    def _target_answer(self, conflict: bool) -> Dict[str, Dict[str, Any]]:
        answer_0 = {'text_id': '3121', 'text': "The staff were ok", 
                    'targets': ['staff'], 'spans': [Span(4, 9)],
                    'target_sentiments': ['neutral'], 
                    'category_sentiments': ['neutral'],
                    'categories': ['service']}
        answer_1 = {'text_id': '2777', 'text': "The day was ok but the food was great", 
                    'targets': ['food'], 'spans': [Span(23, 27)],
                    'target_sentiments': ['positive'], 
                    'category_sentiments': ['positive', 'negative'],
                    'categories': ['food', 'anecdotes/miscellaneous']}
        answer_2 = {'text_id': '2534', 'text': "I had a horrible waiter today", 
                    'category_sentiments': ['negative'],
                    'categories': ['service']}
        answer_3 = {'text_id': '2634', 'text': "Nothing really happened today",
                    'targets': None, 'spans': None, 'target_sentiments': None}
        answer_4 = {'text_id': '1793', 'text': "The drinks were neither good or bad bu the check in was awful", 
                    'targets': ['drinks', 'check'], 'spans': [Span(4, 10), Span(43, 48)],
                    'target_sentiments': ['conflict', 'negative'], 
                    'category_sentiments': ['conflict'],
                    'categories': ['service']}
        if conflict:
            answer_4['target_sentiments'] = ['negative']
            answer_4['spans'] = [Span(43, 48)]
            answer_4['targets'] = ['check']
            answer_4['category_sentiments'] = None
            answer_4['categories'] = None
        return {'3121': answer_0, '2777': answer_1, '2534': answer_2,
                '2634': answer_3, '1793': answer_4}
        

    @pytest.mark.parametrize("conflict", (True, False))
    def test_read_from_file(self, conflict: bool):
        data_fp = Path(self.DATA_PATH_DIR, 'semeval_14_example.xml')
        target_text_collection = semeval_2014(data_fp, conflict)

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
                             'unpassable_semeval_14_example.xml')
        with pytest.raises(ParseError):
            semeval_2014(unreadable_fp, conflict)

    @pytest.mark.parametrize("conflict", (True, False))
    def test_not_semeval_file(self, conflict: bool):
        # Test that it will raise a SyntaxError as the file does not follow 
        # SemEval format.

        unreadable_fp = Path(self.DATA_PATH_DIR, 
                             'not_semeval_14_example.xml')
        with pytest.raises(SyntaxError):
            semeval_2014(unreadable_fp, conflict)

    