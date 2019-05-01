import copy
from itertools import repeat
import traceback
from typing import List, Dict, Any, Tuple, Iterable

import pytest

from target_extraction.data_types import Target_Text, Span

class Test_Target_Text:

    def _exception_examples(self) -> Iterable[Dict[str, Any]]:
        '''
        :returns: The opposite of _passable_examples this returns a list of
                  key word arguments to give to the constructor of 
                  Target_Text the SHOULD raise a ValueError by the 
                  check_list_sizes function.
        '''
        texts = ['The laptop case was great and cover was rubbish'] * 3
        text_ids = ['0', 'another_id', '2']
        possible_spans = [[Span(4, 15)], [Span(30, 35)], 
                          [Span(4, 15), Span(30, 35)]]
        possibe_sentiments = [[0], [1], [0, 1]]
        possible_targets = [['laptop case'], ['cover'], 
                            ['laptop case', 'cover']]
        possible_categories = [['LAPTOP#CASE'], ['LAPTOP'],
                               ['LAPTOP#CASE', 'LAPTOP']]
        # Mismatch in list lengths.
        # Target and Spans length mismatch
        targets_spans = [[possible_targets[0], possible_spans[2]],
                         [possible_targets[2], possible_spans[0]]]
        for target, span in targets_spans:
            yield {'text_id': text_ids[0], 'text': texts[0], 
                   'targets': target, 'spans': span}
        # Sentiment, Categories and spans mismatch
        sentiments_categories_spans = [[possibe_sentiments[0], possible_categories[0], possible_spans[2]],
                                       [possibe_sentiments[0], possible_categories[2], possible_spans[0]],
                                       [possibe_sentiments[2], possible_categories[0], possible_spans[0]]]
        for sentiment, category, span in sentiments_categories_spans:
            yield {'text_id': text_ids[0], 'text': texts[0],
                   'categories': category, 'spans': span, 
                   'sentiments': sentiment}

        # Shouldn't work as the target does not have a reference span
        values = zip(text_ids, texts, possible_targets,
                     possibe_sentiments, possible_categories)
        for _id, text, target, sentiment, category in values:
            yield {'text_id': _id, 'text': text, 'targets': target, 
                   'sentiments': sentiment, 'categories': category}
        # Shouldn't work as the spans and targets do not align, the impossible 
        # Spans are either closer to the target or in-corporate a zero element.
        impossible_spans = [[Span(0, 11)], [Span(31, 35)]]
        span_target_mismatchs = [[possible_spans[0], possible_targets[1]],
                                 [impossible_spans[0], possible_targets[1]],
                                 [impossible_spans[1], possible_targets[1]],
                                 [impossible_spans[0], possible_targets[0]],
                                 [impossible_spans[1], possible_targets[0]],
                                 [possible_spans[1], possible_targets[0]]]
        values = zip(text_ids, texts, span_target_mismatchs)
        for _id, text, span_target in values:
            span, target = span_target
            yield {'text_id': _id, 'text': text, 'spans': span,
                   'targets': target}

    def _passable_examples(self) -> List[Tuple[str, Dict[str, Any]]]:
        '''
        :returns: A list of tuples where the first values is an error messages  
                  the second are key word parameters to give to the constructor 
                  of Target_Text. If the Target_Text cannot be constructed from
                  these parameters then the Error raise from the construction 
                  should be returned with the error message that is associated 
                  to those parameters
        '''
        err_start = 'The check list does not allow through '
        normal_end_err = ' when it should.'
        same_length_end_err = ' when it should as they are of the same lengths'\
                              ' in list size.'
        all_passable_cases = []

        min_case = {'text': 'The laptop case was great and cover was rubbish',
                    'text_id': '0'}
        min_case_err = err_start + 'the minimum fields' + normal_end_err
        all_passable_cases.append((min_case_err, min_case))
        
        one_target = {**min_case, 'targets': ['laptop case'],
                      'spans': [Span(4, 15)]}
        one_target_err = err_start + 'one target and span' + normal_end_err
        all_passable_cases.append((one_target_err, one_target))

        multiple_targets = {**min_case, 'targets': ['laptop case', 'cover'],
                            'spans': [Span(4, 15), Span(30, 35)]}
        multiple_target_err = err_start + 'multiple targets and spans' + \
                              same_length_end_err
        all_passable_cases.append((multiple_target_err, multiple_targets))
        
        one_category = {**min_case, 'categories': ['LAPTOP#CASE']}
        one_category_err = err_start + 'one category' + normal_end_err
        all_passable_cases.append((one_category_err, one_category))

        multiple_categories = {**min_case, 
                               'categories': ['LAPTOP#CASE', 'LAPTOP']}
        multiple_categories_err = err_start + 'multiple categories' + \
                                  normal_end_err
        all_passable_cases.append((multiple_categories_err,
                                   multiple_categories))
        
        one_sentiment = {**min_case, 'sentiments': [0]}
        one_sentiment_err = err_start + 'one sentiment' + normal_end_err
        all_passable_cases.append((one_sentiment_err, one_sentiment))

        multiple_sentiments = {**min_case, 'sentiments': [0, 2]}
        multiple_sentiments_err = err_start + 'multiple sentiments' + \
                                  normal_end_err
        all_passable_cases.append((multiple_sentiments_err,
                                   multiple_sentiments))

        one_sentiment_target = {**one_target, 'sentiments': [0]}
        one_sentiment_target_err = err_start + 'one target and span' + \
                                   same_length_end_err
        all_passable_cases.append((one_sentiment_target_err,
                                   one_sentiment_target))

        multiple_sentiment_targets = {**multiple_targets, 'sentiments': [0, 2]}
        multiple_sentiment_target_err = err_start + \
                                        'multiple targets, spans, sentiments' + \
                                        same_length_end_err
        all_passable_cases.append((multiple_sentiment_target_err,
                                   multiple_sentiment_targets))

        one_category_target = {**one_target, 'categories': ['LAPTOP#CASE']}
        one_category_target_err = err_start + \
                                  'multiple targets, spans, categories' + \
                                  same_length_end_err
        all_passable_cases.append((one_category_target_err, 
                                   one_category_target))

        multiple_category_targets = {**multiple_targets,
                                    'categories': ['LAPTOP#CASE', 'LAPTOP']}
        multiple_category_targets_err = err_start + \
                                        'multiple targets, spans, categories' + \
                                        same_length_end_err
        all_passable_cases.append((multiple_category_targets_err,
                                   multiple_category_targets))

        one_sentiment_category = {**one_sentiment, 
                                  'categories': ['LAPTOP#CASE']}
        one_sentiment_category_err = err_start + \
                                     'one sentiment, category' + \
                                     same_length_end_err
        all_passable_cases.append((one_sentiment_category_err,
                                   one_sentiment_category))

        multiple_sentiment_categories = {**multiple_sentiments,
                                         'categories': ['LAPTOP#CASE', 'LAPTOP']}
        multiple_sentiment_categories_err = err_start + \
                                            'sentiments, categories' + \
                                            same_length_end_err
        all_passable_cases.append((multiple_sentiment_categories_err,
                                   multiple_sentiment_categories))
        
        one_all = {**one_sentiment_category, 'targets': ['laptop case'], 
                   'spans': [Span(4, 15)]}
        one_all_err = err_start + 'one target, span, category, sentiment' + \
                      same_length_end_err
        all_passable_cases.append((one_all_err, one_all))

        multiple_all = {**multiple_sentiment_categories, 
                        'targets': ['laptop case', 'cover'],
                        'spans': [Span(4, 15), Span(30, 35)]}
        multiple_all_err = err_start + \
                           'multiple target, span, category, sentiment' + \
                           same_length_end_err
        all_passable_cases.append((multiple_all_err, multiple_all))
        return all_passable_cases

    def test_list_only(self):
        '''
        Ensures that the assertion is raised if the any of the list only 
        arguments are not lists. This is tested by changing one of the list 
        only arguments to a tuple for each of the possible list only arguments.
        '''
        text = 'The laptop case was great and cover was rubbish'
        text_id = '0'
        span = [Span(4, 15)]
        sentiment = [0]
        target = ['laptop case']
        category = ['LAPTOP#CASE']
        all_list_arguments = [span, sentiment, target, category]
        num_repeats = len(all_list_arguments)
        for index, list_arguments in enumerate(repeat(all_list_arguments, num_repeats)):
            copy_list_arguments = copy.deepcopy(list_arguments)
            copy_list_arguments[index] = tuple(copy_list_arguments[index])

            full_arguments = {'text': text, 'text_id': text_id,
                              'spans': copy_list_arguments[0], 
                              'sentiments': copy_list_arguments[1], 
                              'targets': copy_list_arguments[2], 
                              'categories': copy_list_arguments[3]}
            with pytest.raises(AssertionError):
                Target_Text(**full_arguments)
            
    def test_check_list_sizes(self):
        for passable_err_msg, passable_arguments in self._passable_examples():
            try:
                Target_Text(**passable_arguments)
            except:
                traceback.print_exc()
                raise Exception(passable_err_msg)
        for value_error_arguments in self._exception_examples():
            with pytest.raises(ValueError):
                Target_Text(**value_error_arguments)
        
