import traceback
from typing import List, Dict, Any, Tuple

from target_extraction.data_types import Target_Text, Span

class Test_Target_Text:

    def _passable_examples(self) -> List[Tuple[str, Dict[str, Any]]]:
        '''
        :returns: A list of dictionaries where the keys are error messages and 
                  the values are key word parameters to give to the constructor 
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
            

    def test_check_list(self):
        for passable_err_msg, passable_arguments in self._passable_examples():
            try:
                Target_Text(**passable_arguments)
            except:
                traceback.print_exc()
                raise Exception(passable_err_msg)
        
