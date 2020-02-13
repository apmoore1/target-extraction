import copy
from itertools import repeat
import traceback
from typing import List, Dict, Any, Tuple, Iterable, Callable, Optional

import pytest

from target_extraction.data_types import TargetText, Span, OverLappingTargetsError, AnonymisedError
from target_extraction.tokenizers import spacy_tokenizer, stanford
from target_extraction.pos_taggers import spacy_tagger


class TestTargetText:

    def _regular_examples(self) -> Tuple[List[TargetText],
                                         Tuple[str, List[str], List[List[Span]],
                                               List[int], List[List[str]],
                                               List[List[str]]]]:
        '''
        The first argument `text` in each of the TargetText returned are 
        all the same hence why in the second item in the returned tuple the 
        text argument is not a list.

        :returns: A tuple the first item is a list of TargetText and the 
                  second is a tuple of length 7 containing all of the 
                  arguments that were used to create the TargetTexts.
        '''
        text = 'The laptop case was great and cover was rubbish'
        text_ids = ['0', 'another_id', '2']
        spans = [[Span(4, 15)], [Span(30, 35)], [Span(4, 15), Span(30, 35)]]
        target_sentiments = [[0], ['positive'], [0, 1]]
        targets = [['laptop case'], ['cover'], ['laptop case', 'cover']]
        categories = [['LAPTOP#CASE'], ['LAPTOP'], ['LAPTOP#CASE', 'LAPTOP']]
        category_sentiments = [['pos'],[1],[0, 1]]

        examples = []
        for i in range(3):
            example = TargetText(text, text_ids[i], targets=targets[i],
                                 spans=spans[i], 
                                 target_sentiments=target_sentiments[i],
                                 categories=categories[i],
                                 category_sentiments=category_sentiments[i])
            examples.append(example)
        return examples, (text, text_ids, spans, target_sentiments, 
                          targets, categories, category_sentiments)

    def _exception_examples(self, anonymised: bool = False) -> Iterable[Dict[str, Any]]:
        '''
        :returns: The opposite of _passable_examples this returns a list of
                  key word arguments to give to the constructor of 
                  TargetText that SHOULD raise a ValueError by the 
                  sanitize function.
        '''
        texts = ['The laptop case was great and cover was rubbish'] * 3
        text_ids = ['0', 'another_id', '2']
        possible_spans = [[Span(4, 15)], [Span(30, 35)],
                          [Span(4, 15), Span(30, 35)]]
        possibe_target_sentiments = [[0], [1], [0, 1]]
        possible_targets = [['laptop case'], ['cover'],
                            ['laptop case', 'cover']]
        possible_categories = [['LAPTOP#CASE'], ['LAPTOP'],
                               ['LAPTOP#CASE', 'LAPTOP']]
        possible_category_sentiments = [[0], [1], [0, 1]]
        # Mismatch in list lengths.
        # Target, target sentiments, and Spans length mismatch
        targets_spans = [[possible_targets[0], possible_spans[2], possibe_target_sentiments[0]],
                         [possible_targets[2], possible_spans[0], possibe_target_sentiments[2]],
                         [possible_targets[2], possible_spans[2], possibe_target_sentiments[0]]]
        for target, span, target_sentiment in targets_spans:
            if anonymised:
                yield {'text_id': text_ids[0], 'text': None, 
                       'targets': target, 'spans': span, 
                       'target_sentiments': target_sentiment}
            else:
                yield {'text_id': text_ids[0], 'text': texts[0],
                       'targets': target, 'spans': span, 
                       'target_sentiments': target_sentiment}
            
        # Category and Category sentiments mismatch
        category_category_sentiments = [[possible_categories[0], possible_category_sentiments[2]],
                                        [possible_categories[2], possible_category_sentiments[0]]]
        for categories, category_sentiment in category_category_sentiments:
            if anonymised:
                yield {'text_id': text_ids[0],  'text': None,
                       'category_sentiments': category_sentiment,
                       'categories': categories}
            else:
                yield {'text_id': text_ids[0], 'text': texts[0],
                       'category_sentiments': category_sentiment,
                       'categories': categories}
            

        # Shouldn't work as the target does not have a reference span
        values = zip(text_ids, texts, possible_targets,
                     possibe_target_sentiments, possible_categories,
                     possible_category_sentiments)
        for _id, text, target, target_sentiment, category, category_sentiment in values:
            if anonymised:
                yield {'text_id': _id, 'targets': target, 'text': None,
                       'target_sentiments': target_sentiment, 'categories': category,
                       'category_sentiments': category_sentiment}
            else:
                yield {'text_id': _id, 'text': text, 'targets': target,
                       'target_sentiments': target_sentiment, 'categories': category,
                       'category_sentiments': category_sentiment}
            
        # Shouldn't work as the spans and targets do not align, the impossible
        # Spans are either closer to the target or in-corporate a zero element.
        impossible_spans = [[Span(0, 11)], [Span(31, 35)]]
        span_target_mismatchs = [[possible_spans[0], possible_targets[1]],
                                 [impossible_spans[0], possible_targets[1]],
                                 [impossible_spans[1], possible_targets[1]],
                                 [impossible_spans[0], possible_targets[0]],
                                 [impossible_spans[1], possible_targets[0]],
                                 [possible_spans[1], possible_targets[0]]]
        new_texts = ['The laptop case was great and cover was rubbish'] * len(span_target_mismatchs)
        new_text_ids = [str(i) for i in range(len(span_target_mismatchs))]
        values = zip(new_text_ids, new_texts, span_target_mismatchs)
        for _id, text, span_target in values:
            span, target = span_target
            if anonymised:
                continue
            else:
                yield {'text_id': _id, 'text': text, 'spans': span,
                       'targets': target}
        
        # Shouldn't work when the target is None and the Span is equal to 
        # something, also should not work when the span is (0, 0) and the 
        # target is not None
        span_target_none = [[[Span(0,0)], possible_targets[1]],  
                            [possible_spans[0], [None]]]
        new_texts = ['The laptop case was great and cover was rubbish'] * len(span_target_none)
        new_text_ids = [str(i) for i in range(len(span_target_none))]
        values = zip(new_text_ids, new_texts, span_target_none)
        for _id, text, span_target in values:
            span, target = span_target
            if anonymised:
                yield {'text_id': _id, 'spans': span, 'text': None,
                       'targets': target}
            else:
                yield {'text_id': _id, 'text': text, 'spans': span,
                       'targets': target}
            

    def _passable_examples(self, anonymised: bool = False) -> List[Tuple[str, Dict[str, Any]]]:
        '''
        :param anonymised: Whether the examples should be anonymised examples 
                           which just means removing the text.
        :returns: A list of tuples where the first values is an error messages  
                  the second are key word parameters to give to the constructor 
                  of TargetText. If the TargetText cannot be constructed from
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

        one_sentiment = {**min_case, 'target_sentiments': [0]}
        one_sentiment_err = err_start + 'one sentiment' + normal_end_err
        all_passable_cases.append((one_sentiment_err, one_sentiment))

        multiple_sentiments = {**min_case, 'target_sentiments': [0, 2]}
        multiple_sentiments_err = err_start + 'multiple sentiments' + \
            normal_end_err
        all_passable_cases.append((multiple_sentiments_err,
                                   multiple_sentiments))

        one_sentiment_target = {**one_target, 'target_sentiments': [0]}
        one_sentiment_target_err = err_start + 'one target and span' + \
            same_length_end_err
        all_passable_cases.append((one_sentiment_target_err,
                                   one_sentiment_target))

        multiple_sentiment_targets = {**multiple_targets, 'target_sentiments': [0, 2]}
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

        # Should be able to handle category sentiments
        multiple_all = {**multiple_all, 'category_sentiments': [0, 'pos']}
        all_passable_cases.append(('category sentiments multiple', multiple_all))

        # should be able to handle a different number of categories and 
        # category sentiments compared to the number of targets
        multiple_diff = {**one_all}
        multiple_diff['categories'] = ['LAPTOP', 'CAMERA', 'SCREEN']
        multiple_diff['category_sentiments'] = [0,0,'neg']
        all_passable_cases.append(('category sentiments diff', multiple_diff))

        # Should be able to handle the case where the categories and targets 
        # are somewhat linked and therefore each category has to have a target 
        # even if the target is None value with no reference in the text.
        multiple_all = {**multiple_sentiment_categories,
                        'targets': ['laptop case', 'cover'],
                        'spans': [Span(4, 15), Span(30, 35)]}
        multiple_all['categories'] = ['LAPTOP', 'CAMERA', 'SCREEN']
        multiple_all['category_sentiments'] = [0,0,'neg']
        multiple_all['target_sentiments'] = [0,0,'neg']
        multiple_all['targets'] = ['laptop case', 'cover', None]
        multiple_all['spans'] = [Span(4, 15), Span(30, 35), Span(0,0)]
        all_passable_cases.append(('Target that is Null', multiple_all))
        if anonymised:
            temp_cases = []
            for exception_msg, object_params in all_passable_cases:
                object_params['text'] = None
                temp_cases.append((exception_msg, object_params))
            all_passable_cases = temp_cases
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
        target_sentiment = [0]
        target = ['laptop case']
        category = ['LAPTOP#CASE']
        category_sentiment = ['pos']
        all_list_arguments = [span, target_sentiment, target, category,
                              category_sentiment]
        num_repeats = len(all_list_arguments)
        for index, list_arguments in enumerate(repeat(all_list_arguments, num_repeats)):
            copy_list_arguments = copy.deepcopy(list_arguments)
            copy_list_arguments[index] = tuple(copy_list_arguments[index])

            full_arguments = {'text': text, 'text_id': text_id,
                              'spans': copy_list_arguments[0],
                              'target_sentiments': copy_list_arguments[1],
                              'targets': copy_list_arguments[2],
                              'categories': copy_list_arguments[3],
                              'category_sentiments': copy_list_arguments[4]}
            with pytest.raises(TypeError):
                TargetText(**full_arguments)

    @pytest.mark.parametrize("anonymised", (True, False))
    def test_sanitize(self, anonymised: bool):
        for passable_err_msg, passable_arguments in self._passable_examples(anonymised):
            try:
                if anonymised:
                    passable_arguments['anonymised'] = True
                TargetText(**passable_arguments)
            except:
                raise Exception(passable_err_msg)
        for value_error_arguments in self._exception_examples(anonymised):
            with pytest.raises(ValueError):
                if anonymised:
                    value_error_arguments['anonymised'] = True
                TargetText(**value_error_arguments)
        # Special case only for the anonymised version
        with pytest.raises(ValueError):
            test = TargetText(text_id='1', text=None, anonymised=True)
            test['text'] = 'hello how are you'
            test.sanitize()

    def test_anonymised(self):
        text = "The laptop case was great and cover was rubbish"
        true_target_text = TargetText(text=text, text_id='2', 
                                      targets=["laptop case", "cover", None], 
                                      target_sentiments=[0,1,0],
                                      spans=[Span(4, 15), Span(30, 35), Span(0, 0)],
                                      categories=["LAPTOP#CASE", "LAPTOP"],
                                      category_sentiments=[0, 1])
        # Test the normal case where we want to set the anonymised value to 
        # the same as it was
        assert False == true_target_text.anonymised
        true_target_text.anonymised = False
        assert not true_target_text.anonymised
        assert 'text' in true_target_text
        # Test the case where we want to set it to True
        true_target_text.anonymised = True
        assert not 'text' in true_target_text
        assert true_target_text.anonymised
        
        # Test the case of going from anonymised to de-anonymised
        with pytest.raises(AnonymisedError):
            true_target_text.anonymised = False
        assert true_target_text.anonymised
        # Add the wrong text
        true_target_text._storage['text'] = "The laptop case was great and cove was rubbish"
        with pytest.raises(AnonymisedError):
            true_target_text.anonymised = False
        assert true_target_text.anonymised
        # Add the correct text 
        true_target_text._storage['text'] = text
        true_target_text.anonymised = False
        assert not true_target_text.anonymised
        assert 'text' in true_target_text

    def test_de_anonymise(self):
        true_target_text = TargetText(text='text', text_id='2', 
                                      targets=["laptop case", "cover", None], 
                                      target_sentiments=[0,1,0],
                                      spans=[Span(4, 15), Span(30, 35), Span(0, 0)],
                                      categories=["LAPTOP#CASE", "LAPTOP"],
                                      category_sentiments=[0, 1], anonymised=True)
        # Test the case where the text_id's do not match
        assert true_target_text.anonymised
        assert 'text' not in true_target_text
        text = "The laptop case was great and cover was rubbish"
        new_text = {'text_id': '3', 'text': text}
        with pytest.raises(ValueError):
            true_target_text.de_anonymise(new_text)
        assert true_target_text.anonymised
        assert 'text' not in true_target_text
        # Test the case where it raises an Anonymised Error
        bad_text = "The laptop case was great and cove was rubbish"
        bad_text = {'text_id': '2', 'text': bad_text}
        with pytest.raises(AnonymisedError):
            true_target_text.de_anonymise(bad_text)
        assert true_target_text.anonymised
        assert 'text' not in true_target_text
        # Test the normal case
        new_text['text_id'] = '2'
        true_target_text.de_anonymise(new_text)
        assert not true_target_text.anonymised
        assert text == true_target_text['text']

    def test_force_targets(self):
        # Simple example that nothing should change
        text = 'The laptop case was great and cover was rubbish'
        spans = [Span(4, 15)]
        targets = ['laptop case']
        simple_example = TargetText(text_id='1', spans=spans, text=text, 
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == text
        assert simple_example['spans'] == spans
        assert simple_example['targets'] == targets
        # Simple example with two targets where again nothing should change
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == text
        assert simple_example['spans'] == spans
        assert simple_example['targets'] == targets
        # Edge case example where nothing should change but the targets are at 
        # begining and end of the text.
        spans = [Span(0,3), Span(40,47)]
        targets = ['The', 'rubbish']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == text
        assert simple_example['spans'] == spans
        assert simple_example['targets'] == targets
        # Only the text should change due to the second target linking with a 
        # non-relevant target word.
        text = 'The laptop case was great and coverwas rubbish'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and cover was rubbish'
        assert simple_example['spans'] == spans
        assert simple_example['targets'] == targets
        # The second target spans should change as well as the text
        text = 'The laptop case was great andcover was rubbish'
        spans = [Span(4, 15), Span(29, 34)]
        targets = ['laptop case', 'cover']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and cover was rubbish'
        assert simple_example['spans'] == [Span(4, 15), Span(30, 35)]
        assert simple_example['targets'] == targets
        # The second target spans should change as well as the text
        text = 'The laptop case was great andcoverwas rubbish'
        spans = [Span(4, 15), Span(29, 34)]
        targets = ['laptop case', 'cover']

        # Test the anonymised case
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets, anonymised=True)
        with pytest.raises(AnonymisedError):
            simple_example.force_targets()

        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and cover was rubbish'
        assert simple_example['spans'] == [Span(4, 15), Span(30, 35)]
        assert simple_example['targets'] == targets
        # The second target should change due to the first target changing
        text = 'The laptop casewas great and cover was rubbish'
        spans = [Span(4, 15), Span(29, 34)]
        targets = ['laptop case', 'cover']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and cover was rubbish'
        assert simple_example['spans'] == [Span(4, 15), Span(30, 35)]
        assert simple_example['targets'] == targets

        # All the targets should change
        text = 'Thelaptop casewas great andcoverwas rubbish'
        spans = [Span(3, 14), Span(27, 32)]
        targets = ['laptop case', 'cover']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and cover was rubbish'
        assert simple_example['spans'] == [Span(4, 15), Span(30, 35)]
        assert simple_example['targets'] == targets
        # Edge case with regards to the target being at the end and being
        # changed.
        text = 'Thelaptop casewas great and awfulcover'
        spans = [Span(3, 14), Span(33, 38)]
        targets = ['laptop case', 'cover']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and awful cover'
        assert simple_example['spans'] == [Span(4, 15), Span(36, 41)]
        assert simple_example['targets'] == targets
        # Edge case where two targets are within the same span and changes the 
        # targets based on the changed spans
        text = 'Thelaptop casewas great and awfulcover'
        spans = [Span(3, 14), Span(10, 23)]
        targets = ['laptop case', 'casewas great']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and awfulcover'
        assert simple_example['spans'] == [Span(4, 15), Span(11, 25)]
        assert simple_example['targets'] == ['laptop case', 'case was great']
        # Edge case where two targets are within the same span and changes the 
        # targets based on the changed spans
        text = 'Thelaptop casewas great and awfulcover'
        spans = [Span(10, 23), Span(3, 14)]
        targets = ['casewas great', 'laptop case']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and awfulcover'
        assert simple_example['spans'] == [Span(11, 25), Span(4, 15)]
        assert simple_example['targets'] == ['case was great', 'laptop case']
        # Another example
        text = 'Thelaptopcasewas great and awfulcover'
        spans = [Span(3, 13), Span(9, 22)]
        targets = ['laptopcase', 'casewas great']
        simple_example = TargetText(text_id='1', spans=spans, text=text,
                                    targets=targets)
        simple_example.force_targets()
        assert simple_example['text'] == 'The laptop case was great and awfulcover'
        assert simple_example['spans'] == [Span(4, 15), Span(11, 25)]
        assert simple_example['targets'] == ['laptop case', 'case was great']
        # Test the span that is one from the end case
        text = 'But the machine is awesome and iLife is great and I love Snow Leopard X.'
        spans = [Span(start=31, end=36), Span(start=57, end=71)]
        targets = ['iLife', 'Snow Leopard X']
        end_case = TargetText(text_id='1', spans=spans, text=text,
                              targets=targets)
        end_case.force_targets()
        assert end_case['text'] == 'But the machine is awesome and iLife is great and I love Snow Leopard X .'
        assert end_case['spans'] == [Span(start=31, end=36), Span(start=57, end=71)]
        assert end_case['targets'] == ['iLife', 'Snow Leopard X']
        
    def test_eq(self):
        '''
        Check that the equality between two TargetText is correct
        '''
        # At the moment all the 1's should be the same
        item_1 = TargetText('some text', 'item_1')
        item_1_a = TargetText('some text', 'item_1')
        item_1_b = TargetText('the same', 'item_1')
        # At the moment all items below should not be the same as item_1
        item_2 = TargetText('some text', 'item_2')
        item_3 = TargetText('another text', 'item_3')

        assert item_1 == item_1_a
        assert item_1 == item_1_b

        assert item_1 != item_2
        assert item_1 != item_3

        # Check that it will return False if the two objects are different
        assert item_1 != {'text_id': 'item_1', 'text': 'some text'}

    def test_del(self):
        '''
        Ensure that any key can be deleted as long as it is not a protected 
        key.
        '''
        examples, _ = self._regular_examples()
        protected_keys = examples[0]._protected_keys
        for example in examples:
            for key in list(example.keys()):
                if key not in protected_keys:
                    del example[key]
                    assert key not in example
                else:
                    with pytest.raises(KeyError):
                        del example[key]

    @pytest.mark.parametrize("test_is_list", (True, False))
    @pytest.mark.parametrize("test_list_length", (True, False))
    def test_set(self, test_is_list: bool, test_list_length: bool):
        '''
        Ensure that any key can be changed as long as it is not a protected 
        key and that it follows the sanitize checks.
        '''
        examples, _ = self._regular_examples()
        protected_keys = examples[0]._protected_keys
        for example in examples:
            for key in list(example.keys()):
                new_value = 'some value'
                if key not in protected_keys:
                    current_value = example[key]
                    if isinstance(current_value, list):
                        if test_is_list:
                            with pytest.raises(TypeError):
                                example[key] = new_value
                        elif test_list_length:
                            new_value = [new_value] * (len(current_value) + 1)
                            with pytest.raises(ValueError):
                                example[key] = new_value
                        else:
                            new_value = [new_value] * len(current_value)
                            example[key] = new_value
                    else:
                        example[key] = new_value
                else:
                    with pytest.raises(KeyError):
                        example[key] = new_value

    def test_get_item(self):
        '''
        Ensure that the get item works.
        '''
        examples, example_arguments = self._regular_examples()
        argument_names = ['text', 'text_id', 'spans', 'target_sentiments',
                          'targets', 'categories', 'category_sentiments']
        for example_index, example in enumerate(examples):
            for argument_index, arguments in enumerate(example_arguments):
                if argument_index == 0:
                    argument = arguments
                else:
                    argument = arguments[example_index]
                argument_name = argument_names[argument_index]
                assert example[argument_name] == argument

    def test_length(self):
        examples, _ = self._regular_examples()
        example = examples[0]
        assert len(example) == 7
        del example['target_sentiments']
        assert len(example) == 6
    
    def test_to_json(self):
        true_json_text = ('{"text": "The laptop case was great and cover was rubbish", '
                          '"text_id": "2", "targets": ["laptop case", "cover"], '
                          '"spans": [[4, 15], [30, 35]], "target_sentiments": [0, 1], '
                          '"categories": ["LAPTOP#CASE", "LAPTOP"], "category_sentiments": [0, 1]}')
        examples, _ = self._regular_examples()
        example = examples[-1]
        assert example.to_json() == true_json_text
        # Test it when a target is of Value None
        true_json_text = ('{"text": "The laptop case was great and cover was rubbish", '
                          '"text_id": "2", "targets": ["laptop case", "cover", null], '
                          '"spans": [[4, 15], [30, 35], [0, 0]], "target_sentiments": [0, 1, 0], '
                          '"categories": ["LAPTOP#CASE", "LAPTOP"], "category_sentiments": [0, 1]}')
        text = "The laptop case was great and cover was rubbish"
        true_target_text = TargetText(text=text, text_id='2', 
                                      targets=["laptop case", "cover", None], 
                                      target_sentiments=[0,1,0],
                                      spans=[Span(4, 15), Span(30, 35), Span(0, 0)],
                                      categories=["LAPTOP#CASE", "LAPTOP"],
                                      category_sentiments=[0, 1])
        assert true_target_text.to_json() == true_json_text

        # anonymised case (The JSON version should not have any text)
        true_json_anonymised_text = ('{"text_id": "2", "targets": ["laptop case", "cover", null], '
                          '"spans": [[4, 15], [30, 35], [0, 0]], "target_sentiments": [0, 1, 0], '
                          '"categories": ["LAPTOP#CASE", "LAPTOP"], "category_sentiments": [0, 1]}')
        text = "The laptop case was great and cover was rubbish"
        true_target_text = TargetText(text=text, text_id='2', 
                                      targets=["laptop case", "cover", None], 
                                      target_sentiments=[0,1,0],
                                      spans=[Span(4, 15), Span(30, 35), Span(0, 0)],
                                      categories=["LAPTOP#CASE", "LAPTOP"],
                                      category_sentiments=[0, 1], anonymised=True)
        assert true_target_text.to_json() == true_json_anonymised_text

    def test_from_json(self):
        json_text = ('{"text": "The laptop case was great and cover was rubbish", '
                     '"text_id": "2", "targets": ["laptop case", "cover", null], '
                     '"spans": [[4, 15], [30, 35], [0,0]], "target_sentiments": [0, 1, 0], '
                     '"categories": ["LAPTOP#CASE", "LAPTOP"], "category_sentiments": [0, 1]}')
        example_from_json = TargetText.from_json(json_text)
        example_spans: List[Span] = example_from_json['spans']
        for span in example_spans:
            assert isinstance(span, Span), f'{span} should be of type Span'
        
        text = "The laptop case was great and cover was rubbish"
        true_target_text = TargetText(text=text, text_id='2', 
                                      targets=["laptop case", "cover", None], 
                                      target_sentiments=[0,1,0],
                                      spans=[Span(4, 15), Span(30, 35), Span(0, 0)],
                                      categories=["LAPTOP#CASE", "LAPTOP"],
                                      category_sentiments=[0, 1])
        for key, value in true_target_text.items():
            assert value == example_from_json[key]
        # Where all but the core values are None
        json_text = ('{"text": "anything", "text_id": "1", "targets": null, '
                     '"spans": null, "target_sentiments": null, '
                     '"categories": null, "category_sentiments": null}')
        example_from_json = TargetText.from_json(json_text)
        correct_answer = TargetText(text='anything', text_id='1')
        for key, value in correct_answer.items():
            assert value == example_from_json[key]
        
        # Loading when it contains a field that is not in the TargetText
        # constructor.
        json_text = ('{"text": "anything", "text_id": "1", "targets": null, '
                     '"spans": null, "target_sentiments": null, '
                     '"categories": null, "category_sentiments": null,'
                     '"tokenized_text": ["anything"]}')
        example_from_json = TargetText.from_json(json_text)
        correct_answer = TargetText(text='anything', text_id='1')
        correct_answer["tokenized_text"] = ["anything"]
        for key, value in correct_answer.items():
            assert value == example_from_json[key]
        # Ensure that a KeyError is raised if `text` or `text_id` is not 
        # in the json text
        bad_json_text_0 = ('{"text": "anything"}')
        bad_json_text_1 = ('{"text_id": "1"}')
        bad_json_text_2 = ('{}')
        good_json_text = ('{"text": "anything", "text_id": "1"}')
        with pytest.raises(KeyError):
            TargetText.from_json(bad_json_text_0)
        with pytest.raises(KeyError):
            TargetText.from_json(bad_json_text_1)
        with pytest.raises(KeyError):
            TargetText.from_json(bad_json_text_2)
        TargetText.from_json(good_json_text)
        # Should raise a ValueError through sanitize
        bad_json_text_3 = ('{"text": "anything", "spans": [[0,3]], '
                           '"targets": ["an"], "text_id": "1"}')
        with pytest.raises(ValueError):
            TargetText.from_json(bad_json_text_3)
        
        # Test the anonymised cases
        json_text = ('{"text": "The laptop case was great and cover was rubbish", '
                     '"text_id": "2", "targets": ["laptop case", "cover", null], '
                     '"spans": [[4, 15], [30, 35], [0,0]], "target_sentiments": [0, 1, 0], '
                     '"categories": ["LAPTOP#CASE", "LAPTOP"], "category_sentiments": [0, 1]}')
        example_from_json = TargetText.from_json(json_text, anonymised=True)
        example_spans: List[Span] = example_from_json['spans']
        for span in example_spans:
            assert isinstance(span, Span), f'{span} should be of type Span'
        
        true_target_text = TargetText(text=None, text_id='2', 
                                      targets=["laptop case", "cover", None], 
                                      target_sentiments=[0,1,0],
                                      spans=[Span(4, 15), Span(30, 35), Span(0, 0)],
                                      categories=["LAPTOP#CASE", "LAPTOP"],
                                      category_sentiments=[0, 1], anonymised=True)
        for key, value in true_target_text.items():
            assert value == example_from_json[key]
        assert example_from_json.anonymised == True
        

    @pytest.mark.parametrize("tokenizer", (str.split, spacy_tokenizer()))
    @pytest.mark.parametrize("type_checks", (True, False))
    def test_tokenize(self, tokenizer: Callable[[str], List[str]], 
                      type_checks: bool):
        def not_char_preserving_tokenizer(text: str) -> List[str]:
            tokens = text.split()
            alt_tokens = []
            for token in tokens:
                if token == 'laptop':
                    alt_tokens.append('latop')
                else:
                    alt_tokens.append(token)
            return alt_tokens
        # Test the normal case with one TargetText Instance in the collection
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        test_target_text = TargetText(text=text, text_id=text_id)
        test_target_text.tokenize(tokenizer, perform_type_checks=type_checks)
        tokenized_answer = ['The', 'laptop', 'case', 'was', 'great', 'and',
                            'cover', 'was', 'rubbish']
        assert test_target_text['tokenized_text'] == tokenized_answer
        # Test the anonymised case
        test_target_text = TargetText(text=text, text_id=text_id, anonymised=True)
        with pytest.raises(AnonymisedError):
            test_target_text.tokenize(tokenizer, perform_type_checks=type_checks)

        # Test the case where the tokenizer function given does not return a
        # List
        test_target_text = TargetText(text=text, text_id=text_id)
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.tokenize(str.strip, perform_type_checks=type_checks)
        else:
            test_target_text.tokenize(str.strip, perform_type_checks=type_checks)
        # Test the case where the tokenizer function given returns a list but
        # not a list of strings
        test_target_text = TargetText(text=text, text_id=text_id)
        def token_len(text): return [len(token) for token in text.split()]
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.tokenize(token_len, perform_type_checks=type_checks)

        # Test the case where the TargetTextCollection contains instances
        # but the instances have no text therefore returns a ValueError
        test_target_text = TargetText(text='', text_id=text_id)
        with pytest.raises(ValueError):
            test_target_text.tokenize(tokenizer, perform_type_checks=type_checks)
        # Test the case when the tokenizer is not character preserving
        test_target_text = TargetText(text=text, text_id=text_id)
        with pytest.raises(ValueError):
            test_target_text.tokenize(not_char_preserving_tokenizer, perform_type_checks=type_checks)

    @pytest.mark.parametrize("type_checks", (True, False))
    def test_pos_text(self, type_checks: bool):
        # Test the normal case
        text = "The laptop's case was great and cover was rubbish"
        text_id = '2'
        pos_tagger = spacy_tagger()
        test_target_text = TargetText(text=text, text_id=text_id)
        test_target_text.pos_text(pos_tagger, perform_type_checks=type_checks)

        pos_answer = ['DET', 'NOUN', 'PART', 'NOUN', 'VERB', 'ADJ', 'CCONJ', 
                      'NOUN', 'VERB', 'ADJ']
        tok_answer = ['The', 'laptop', "'s", 'case', 'was', 'great', 'and', 
                      'cover', 'was', 'rubbish']
        assert test_target_text['pos_tags'] == pos_answer
        assert test_target_text['tokenized_text'] == tok_answer 
        # Test the case where the return is not a tuple
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.pos_text(str.strip, perform_type_checks=type_checks)
        def false_tuple_tagger(text):
            return (1, 2, 3)
        # Test the case where the return is not a Tuple of length two
        if type_checks:
            with pytest.raises(ValueError):
                test_target_text.pos_text(false_tuple_tagger, perform_type_checks=type_checks)
        def tuple_but_not_lists(text):
            return (1, 2)
        # Test the case that the tagger returns a tuple of length 2 but 
        # the tuple contains no lists
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.pos_text(tuple_but_not_lists, perform_type_checks=type_checks)
        # Test the case where the first value in the tuple is a list but the 
        # second is not
        def tuple_but_not_all_lists(text):
            return ([1], 2)
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.pos_text(tuple_but_not_all_lists, perform_type_checks=type_checks)
        # Test the case where the tagger function given returns a list but
        # not a list of strings
        def list_not_strings(text):
            return ([1, 2], [3, 4])
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.pos_text(list_not_strings, perform_type_checks=type_checks)
        # Test the case where the first list are strings but the second is not
        def list_but_one_strings(text):
            return (['1', '2'], [3, 4])
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.pos_text(list_but_one_strings, perform_type_checks=type_checks)
        # Test the case where the pos tags produced are zero length
        def tokens_but_no_pos(text):
            return (['1', '2'], [])
        with pytest.raises(ValueError):
            test_target_text.pos_text(tokens_but_no_pos, perform_type_checks=type_checks)
        # Test the case where there are not the same number of pos as tokens
        def not_same_length(text):
            return (['1', '2'], ['2'])
        with pytest.raises(ValueError):
            test_target_text.pos_text(not_same_length, perform_type_checks=type_checks)
        # Test that the tokens produced by the POS tagger overwrite any new 
        # tokens
        test_target_text.pos_text(pos_tagger, perform_type_checks=type_checks)
        test_target_text.tokenize(str.split)
        split_tokens = ['The', "laptop's", 'case', 'was', 'great', 'and', 
                         'cover', 'was', 'rubbish']
        assert test_target_text['tokenized_text'] == split_tokens
        test_target_text.pos_text(pos_tagger, perform_type_checks=type_checks)
        assert test_target_text['tokenized_text'] == tok_answer
        
        # Test the anonymised case
        test_target_text = TargetText(text=text, text_id=text_id, anonymised=True)
        with pytest.raises(AnonymisedError):
            test_target_text.pos_text(pos_tagger, perform_type_checks=type_checks)

    @pytest.mark.parametrize("label_key", (None, 'target_sentiments'))
    @pytest.mark.parametrize("per_target", (True, False))
    def test_sequence_labels(self, per_target: bool, label_key: str):
        # Test that it will raise a KeyError if it has not been tokenized
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        target_sentiments = ['POS', 'NEG']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets, target_sentiments=target_sentiments)
        with pytest.raises(KeyError):
            test.sequence_labels(per_target, label_key=label_key)
        # Test the case where there are no spans or targets
        test = TargetText(text=text, text_id=text_id, 
                          target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        if per_target:
            answer = [answer]
        assert answer == test['sequence_labels']
        # Test the basic case where we have only one target of one word
        text = 'The laptop'
        spans = [Span(4, 10)]
        targets = ['laptop']
        target_sentiments = ['POS']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets, target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'B']
        if label_key is not None:
            answer = ['O', 'B-POS']
        if per_target:
            answer = [answer]
        assert answer == test['sequence_labels']
        # Test the case where the sequence is greater more than one word
        text = 'The laptop case'
        spans = [Span(4, 15)]
        targets = ['laptop case']
        target_sentiments = ['NEG']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets, target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'B', 'I']
        if label_key is not None:
            answer = ['O', 'B-NEG', 'I-NEG']
        if per_target:
            answer = [answer]
        assert answer == test['sequence_labels']
        # Test the case where the target is the first word
        text = 'laptop case is great'
        spans = [Span(0, 11)]
        targets = ['laptop case']
        target_sentiments = ['NEU']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets, target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['B', 'I', 'O', 'O']
        if label_key is not None:
            answer = ['B-NEU', 'I-NEU', 'O', 'O']
        if per_target:
            answer = [answer]
        assert answer == test['sequence_labels']
        # Test the case where there is more than one target
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        target_sentiments = ['POS', 'NEG']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets, target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'B', 'I', 'O', 'O', 'O', 'B', 'O', 'O']
        if label_key is not None:
            answer = ['O', 'B-POS', 'I-POS', 'O', 'O', 'O', 'B-NEG', 'O', 'O']
        if per_target:
            answer = [['O', 'B', 'I', 'O', 'O', 'O', 'O', 'O', 'O'], 
                      ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O']]
            if label_key is not None:
                answer = [['O', 'B-POS', 'I-POS', 'O', 'O', 'O', 'O', 'O', 'O'], 
                          ['O', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'O', 'O']]
        assert answer == test['sequence_labels']
        # Test the case where the tokens do not perfectly align the target text 
        # spans and therefore only match on one of the perfectly aligned tokens
        text = 'The laptop;priced was high and the laptop cover-ed was based'
        text_id = '2'
        spans = [Span(11, 17), Span(35, 47)]
        targets = ['priced', 'laptop cover']
        target_sentiments = ['NEU', 'NEG']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets, target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O']
        if label_key is not None:
            answer = ['O', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'O', 'O', 'O']
        if per_target:
            answer = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
                      ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O']]
            if label_key is not None:
                answer = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
                          ['O', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'O', 'O', 'O']]
        assert answer == test['sequence_labels']
        # Different tokenizer would get a different result
        test.tokenize(spacy_tokenizer())
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O', 'O']
        if label_key is not None:
            answer = ['O', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'I-NEG', 'O', 'O', 'O', 'O']
        if per_target:
            answer = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
                      ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O', 'O']]
            if label_key is not None:
                answer = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
                          ['O', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'I-NEG', 'O', 'O', 'O', 'O']]
        assert answer == test['sequence_labels']
        test.tokenize(stanford())
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O', 'O']
        if label_key is not None:
            answer = ['O', 'O', 'O', 'B-NEU', 'O', 'O', 'O', 'O', 'B-NEG', 'I-NEG', 'O', 'O', 'O', 'O']
        if per_target:
            answer = [['O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
                      ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O', 'O']]
            if label_key is not None:
                answer = [['O', 'O', 'O', 'B-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
                          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'I-NEG', 'O', 'O', 'O', 'O']]
            
        assert answer == test['sequence_labels']
        # Test if using force targets will correct the above mistakes which it
        # should
        test.force_targets()
        assert test['text'] == 'The laptop; priced was high and the laptop cover -ed was based'
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'O', 'B', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O']
        if label_key is not None:
            answer = ['O', 'O', 'B-NEU', 'O', 'O', 'O', 'O', 'B-NEG', 'I-NEG', 'O', 'O', 'O']
        if per_target:
            answer = [['O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
                      ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O']]
            if label_key is not None:
                answer = [['O', 'O', 'B-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
                          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'I-NEG', 'O', 'O', 'O']]
        assert answer == test['sequence_labels']
        
        # Case where two targets are next to each other.
        text = 'The laptop case price was great and bad'
        spans = [Span(4, 15), Span(16, 21)]
        targets = ['laptop case', 'price']
        target_sentiments = ['NEU', 'NEG']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets, target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'B', 'I', 'B', 'O', 'O', 'O', 'O']
        if label_key is not None:
            answer = ['O', 'B-NEU', 'I-NEU', 'B-NEG', 'O', 'O', 'O', 'O']
        if per_target:
            answer = [['O', 'B', 'I', 'O', 'O', 'O', 'O', 'O'], 
                      ['O', 'O', 'O', 'B', 'O', 'O', 'O', 'O']]
            if label_key is not None:
                answer = [['O', 'B-NEU', 'I-NEU', 'O', 'O', 'O', 'O', 'O'], 
                          ['O', 'O', 'O', 'B-NEG', 'O', 'O', 'O', 'O']]
        assert answer == test['sequence_labels']

        # Case where the targets are at the start and end and contain no `O`
        # Handle cases at the extreme of the sentence (beginning and end)
        text = 'Laptop priced'
        spans = [Span(0, 6), Span(7, 13)]
        targets = ['Laptop', 'priced']
        target_sentiments = ['NEU', 'NEU']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets, target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['B', 'B']
        if label_key is not None:
            answer = ['B-NEU', 'B-NEU']
        if per_target:
            answer = [['B', 'O'], ['O', 'B']]
            if label_key is not None:
                answer = [['B-NEU', 'O'], ['O', 'B-NEU']]
        assert answer == test['sequence_labels']
        
        # Handle the case where the targets are next to each other where one 
        # ends with an I tag and the other starts with a B tag.
        text = 'Laptop cover priced'
        spans = [Span(0, 12), Span(13, 19)]
        targets = ['Laptop cover', 'priced']
        target_sentiments = ['POS', 'NEG']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets, target_sentiments=target_sentiments)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['B', 'I', 'B']
        if label_key is not None:
            answer = ['B-POS', 'I-POS', 'B-NEG']
        if per_target:
            answer = [['B', 'I', 'O'], ['O', 'O', 'B']]
            if label_key is not None:
                answer = [['B-POS', 'I-POS', 'O'], ['O', 'O', 'B-NEG']]
        assert answer == test['sequence_labels']

        # Handle the case where the spans order is the same as target but if 
        # sorted by span index it would not.
        text = 'Laptop cover priced'
        spans = [Span(13, 19), Span(0, 12)]
        targets = ['priced', 'Laptop cover']
        target_sentiments = ['POS', 'POS']
        target_categories_error = ['Nothing']
        labels_error = None
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets, target_sentiments=target_sentiments,
                          target_categories_error=target_categories_error,
                          labels_error=labels_error)
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['B', 'I', 'B']
        if label_key is not None:
            answer = ['B-POS', 'I-POS', 'B-POS']
        if per_target:
            answer = [['O', 'O', 'B'], ['B', 'I', 'O']]
            if label_key is not None:
                answer = [['O', 'O', 'B-POS'], ['B-POS', 'I-POS', 'O']]
        assert answer == test['sequence_labels']

        #
        # Label Key specific errors
        #
        # Raise KeyError is the label key does not exist
        with pytest.raises(KeyError):
            test.sequence_labels(per_target, label_key='error')
        # Raises ValueError when the label key does not the same number of values 
        # as the number of targets
        with pytest.raises(ValueError):
            test.sequence_labels(per_target, label_key='target_categories_error')
        # Raises TypeError when the label key is not a list
        with pytest.raises(TypeError):
            test.sequence_labels(per_target, label_key='labels_error')

        
        # Handle the case where there are no sequence labels
        test = TargetText(text=text, text_id='1')
        test.tokenize(str.split)
        test.sequence_labels(per_target, label_key=label_key)
        answer = ['O', 'O', 'O']
        if per_target:
            answer = [['O', 'O', 'O']]
        assert answer == test['sequence_labels']

        # Test the anonymised case
        test_target_text = TargetText(text=text, text_id='1', anonymised=True)
        with pytest.raises(AnonymisedError):
            test_target_text.sequence_labels(per_target, label_key=label_key)
        

    def test_key_error(self):
        # Simple example that should pass
        example = TargetText(text='some text', text_id='2')
        example._key_error('text')
        # Simple example that should raise a KeyError
        with pytest.raises(KeyError):
            example._key_error('tokenized_text')
        # Proof that it should now not raise a KeyError
        example.tokenize(str.split)
        example._key_error('tokenized_text')

    def test_get_sequence_indexs(self):
        # Standard example with two targets
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        correct_sequence_indexs = [[1, 2], [6]]
        assert correct_sequence_indexs == sequence_indexs
        
        #
        # Test all raise error cases
        #
        # Test that it will raise a KeyError if a key that does not exist is 
        # provided
        with pytest.raises(KeyError):
            test.get_sequence_spans('predicted_sequence_labels')
        # Test that it will raise a Value Error when the number of tokens is 
        # not equal to the number of sequence labels
        value_error_target = TargetText(text=text, text_id=text_id, spans=spans, 
                                        targets=targets)
        value_error_target.tokenize(str.split)
        value_error_target.sequence_labels()
        error_tokens = ['The', 'lap', 'top', 'case', 'was', 'great', 'and', 
                        'cover', 'was', 'rubbish']
        value_error_target._storage['tokenized_text'] = error_tokens 
        with pytest.raises(ValueError):
            value_error_target.get_sequence_indexs('sequence_labels')
        # Ensure that it raises an error if the sequence tags are not BIO
        test['sequence_labels'] = ['O', 'B', 'E', 'I', 'O', 'O', 'O', 'B', 'E']
        with pytest.raises(ValueError):
            test.get_sequence_spans('sequence_labels')

        # Test the anonymised case
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets, anonymised=True)
        with pytest.raises(AnonymisedError):
            test.get_sequence_indexs('sequence_labels')

        #
        # More sophisticated normal cases
        #

        # Case where two targets are next to each other.
        text = 'The laptop case price was great and bad'
        spans = [Span(4, 15), Span(16, 21)]
        targets = ['laptop case', 'price']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        correct_sequence_indexs = [[1, 2], [3]]
        assert correct_sequence_indexs == sequence_indexs

        # Case where the text and the tokens do not perfectly align with 
        # each other
        text = 'The laptop;priced was high and the laptop cover-ed was based'
        text_id = '2'
        spans = [Span(11, 17), Span(35, 47)]
        targets = ['priced', 'laptop cover']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        assert [[6]] == sequence_indexs
        
        # Using a tokenizer that can perfectly seperate the targets out
        test.tokenize(stanford())
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        assert [[3], [8, 9]] == sequence_indexs

        # Handle cases at the extreme of the sentence (beginning and end)
        text = 'Laptop was ;priced'
        spans = [Span(0, 6), Span(12, 18)]
        targets = ['Laptop', 'priced']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        assert [[0]] == sequence_indexs
        # Perfect tokenizer
        test.tokenize(stanford())
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        assert [[0], [3]] == sequence_indexs

        # Handle multi word targets greater than 2
        text = 'The Laptop cover price was very good'
        spans = [Span(4, 22)]
        targets = ['Laptop cover price']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        assert [[1,2,3]] == sequence_indexs

        # Handle the case where the targets are next to each other and both 
        # start with B tags.
        text = 'Laptop priced'
        spans = [Span(0, 6), Span(7, 13)]
        targets = ['Laptop', 'priced']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        assert [[0], [1]] == sequence_indexs

        # Handle the case where the targets are next to each other where one 
        # ends with an I tag and the other starts with a B tag.
        text = 'Laptop cover priced'
        spans = [Span(0, 12), Span(13, 19)]
        targets = ['Laptop cover', 'priced']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        assert [[0, 1], [2]] == sequence_indexs

        # Test the case where there are no sequence spans
        test = TargetText(text=text, text_id='1')
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_indexs = test.get_sequence_indexs('sequence_labels')
        assert [] == sequence_indexs

    @pytest.mark.parametrize("confidence", (None, 0.9))
    def test_get_sequence_spans(self, confidence: float):
        # Simple example where the text and the tokens are perfectly aligned.
        # Also it is an example that includes single target and multi word 
        # target
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == spans

        # Test that it will raise a KeyError if a key that does not exist is 
        # provided
        with pytest.raises(KeyError):
            test.get_sequence_spans('predicted_sequence_labels')
        # Test the anonymised case
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets, anonymised=True)
        with pytest.raises(AnonymisedError):
            test.get_sequence_spans('sequence_labels')
        
        # Case where two targets are next to each other.
        text = 'The laptop case price was great and bad'
        spans = [Span(4, 15), Span(16, 21)]
        targets = ['laptop case', 'price']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == spans

        # Case where the text and the tokens do not perfectly align with 
        # each other
        text = 'The laptop;priced was high and the laptop cover-ed was based'
        text_id = '2'
        spans = [Span(11, 17), Span(35, 47)]
        targets = ['priced', 'laptop cover']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == [Span(35, 41)]
        # Using a tokenizer that can perfectly seperate the targets out
        test.tokenize(stanford())
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == spans

        # Handle cases at the extreme of the sentence (beginning and end)
        text = 'Laptop was ;priced'
        spans = [Span(0, 6), Span(12, 18)]
        targets = ['Laptop', 'priced']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == [Span(0, 6)]
        # Perfect tokenizer
        test.tokenize(stanford())
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == spans

        # Handle multi word targets greater than 2
        text = 'The Laptop cover price was very good'
        spans = [Span(4, 22)]
        targets = ['Laptop cover price']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == spans

        # Ensure that it raises an error if the sequence tags are not BIO
        test['sequence_labels'] = ['O', 'B', 'E', 'I', 'O', 'O', 'O']
        with pytest.raises(ValueError):
            test.get_sequence_spans('sequence_labels')

        # Handle the case where the targets are next to each other and both 
        # start with B tags.
        text = 'Laptop priced'
        spans = [Span(0, 6), Span(7, 13)]
        targets = ['Laptop', 'priced']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == spans

        # Handle the case where the targets are next to each other where one 
        # ends with an I tag and the other starts with a B tag.
        text = 'Laptop cover priced'
        spans = [Span(0, 12), Span(13, 19)]
        targets = ['Laptop cover', 'priced']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert sequence_spans == spans

        # Test the case where there are no sequence spans
        test = TargetText(text=text, text_id='1')
        test.tokenize(str.split)
        test.sequence_labels()
        sequence_spans = test.get_sequence_spans('sequence_labels')
        assert [] == sequence_spans

        #
        # Testing the confidence argument
        #

        # Simple case where the confidence values should not affect the 
        # output
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        confidences = [0.0, 0.91, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        sequence_labels = ['O', 'B', 'I', 'O', 'O', 'O', 'B', 'O', 'O']
        test = TargetText(text=text, text_id=text_id, 
                          sequence_labels=sequence_labels, confidence=confidences)
        test.tokenize(str.split)
        test_spans = test.get_sequence_spans('sequence_labels', 
                                             confidence=confidence)
        assert [Span(4, 15), Span(30, 35)] == test_spans
        # The last sequence label should not be allowed in the confidence case
        confidences = [0.0, 0.91, 1.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]
        test['confidence'] = confidences
        test_spans = test.get_sequence_spans('sequence_labels', 
                                             confidence=confidence)
        if confidence is not None:
            assert [Span(4, 15)] == test_spans
        else:
            assert [Span(4, 15), Span(30, 35)] == test_spans
        # The first set of spans should not exist as all of the label are 
        # below the confidence threshold
        confidences = [0.0, 0.89, 0.0, 0.0, 0.0, 0.0, 0.91, 0.0, 0.0]
        test['confidence'] = confidences
        test_spans = test.get_sequence_spans('sequence_labels', 
                                             confidence=confidence)
        if confidence is not None:
            assert [Span(30, 35)] == test_spans
        else:
            assert [Span(4, 15), Span(30, 35)] == test_spans
        # The first set of spans should not exist as at least one of the labels
        # is below the confidence threshold
        confidences = [0.0, 0.9, 0.91, 0.0, 0.0, 0.0, 0.91, 0.0, 0.0]
        test['confidence'] = confidences
        test_spans = test.get_sequence_spans('sequence_labels', 
                                             confidence=confidence)
        if confidence is not None:
            assert [Span(30, 35)] == test_spans
        else:
            assert [Span(4, 15), Span(30, 35)] == test_spans
        # None of the spans should exist as they are all below the confidence 
        # threshold
        confidences = [0.0, 0.89, 0.89, 0.0, 0.0, 0.0, 0.51, 0.0, 0.0]
        test['confidence'] = confidences
        test_spans = test.get_sequence_spans('sequence_labels', 
                                             confidence=confidence)
        if confidence is not None:
            assert [] == test_spans
        else:
            assert [Span(4, 15), Span(30, 35)] == test_spans
        # Test the confidence ValueError case
        with pytest.raises(ValueError):
            test.get_sequence_spans('sequence_labels', confidence=1.01)
        with pytest.raises(ValueError):
            test.get_sequence_spans('sequence_labels', confidence=-0.01)
        # Test the KeyError test for confidence
        del test['confidence']
        if confidence is not None:
            with pytest.raises(KeyError):
                test.get_sequence_spans('sequence_labels', 
                                        confidence=confidence)

    @pytest.mark.parametrize("remove_empty", (False, True))
    def test_one_sample_per_span(self, remove_empty: bool):
        # Case where nothing should change.
        examples, _ = self._regular_examples()
        for example in examples:
            one_span_per_sample = example.one_sample_per_span(remove_empty=remove_empty)
            for key, value in example.items():
                if key in ['target_sentiments', 'categories', 
                           'category_sentiments']:
                    assert one_span_per_sample[key] == None
                else:
                    assert value == one_span_per_sample[key]
        # Case where we have one span twice
        example = examples[0]
        example._storage['spans'] = [Span(4,15), Span(4,15)]
        example._storage['targets'] = ['laptop case', 'laptop case']
        example._storage['target_sentiments'] = [0,1]
        example._storage['categories'] = ['laptop', 'case']
        example._storage['category_sentiments'] = [1, 0]
        assert example['spans'] == [Span(4,15), Span(4,15)]
        example_one_span = example.one_sample_per_span(remove_empty=remove_empty)

        correct_answers = {'spans': [Span(4,15)], 'targets': ['laptop case'],
                           'target_sentiments': None, 'categories': None,
                           'category_sentiments': None}
        for key, value in correct_answers.items():
            assert value == example_one_span[key]
        
        # Case where we have multiple different spans and not all need 
        # removing. We also check that the original TargetText has not been 
        # changed
        example = examples[0]
        example._storage['spans'] = [Span(36, 39), Span(4,15), Span(30, 35),  
                                     Span(4,15), Span(0, 0), Span(0, 0), Span(30, 35)]
        example._storage['targets'] = ['was', 'laptop case', 'cover', 
                                       'laptop case', None, None, 'cover']
        example._storage['target_sentiments'] = [0,1,0,1,1,0,0]
        example._storage['categories'] = ['nothing', 'laptop', 'case', 'laptop', 
                                          'another', 'empty', 'case']
        example._storage['category_sentiments'] = [0,1,0,1,0,1,1]
        assert example['spans'] == [Span(36, 39), Span(4,15), Span(30, 35),  
                                    Span(4,15), Span(0, 0), Span(0, 0), Span(30, 35)]
        example_one_span = example.one_sample_per_span(remove_empty=remove_empty)
        # Check original has not been changed
        assert example['spans'] == [Span(36, 39), Span(4,15), Span(30, 35),  
                                    Span(4,15), Span(0, 0), Span(0, 0), Span(30, 35)]

        correct_answers = {'spans': [Span(0, 0), Span(4, 15), Span(30, 35), Span(36, 39)], 
                           'targets': ['', 'laptop case', 'cover', 'was'],
                           'target_sentiments': None, 'categories': None,
                           'category_sentiments': None}
        if remove_empty:
            correct_answers['spans'] = [Span(4, 15), Span(30, 35), Span(36, 39)]
            correct_answers['targets'] = ['laptop case', 'cover', 'was']
        for key, value in correct_answers.items():
            assert value == example_one_span[key]
        
        # Check that it does return a TargetText instance
        assert isinstance(example_one_span, TargetText)

        # Chack that it can handle the case where there are no targets or spans
        edge_example = TargetText(text='here is some text', text_id='1')
        edge_test = edge_example.one_sample_per_span(remove_empty=remove_empty)
        assert edge_test['text'] == 'here is some text'
        assert edge_test['text_id'] == '1'

        # Test the anonymised case
        test = TargetText(text='here is some text', text_id='1', anonymised=True)
        with pytest.raises(AnonymisedError):
            test.one_sample_per_span(remove_empty=remove_empty)

    def test_constructior_additional_data(self):
        # Want to test the normal case of just adding unrelevant data
        normal_kwargs = {'text': 'This is some text', 'text_id': '1',
                         'pos_tags': ['NN'], 
                         'tokenized_text': ['This', 'is', 'some', 'text']}
        test = TargetText(**normal_kwargs)
        for key, value in normal_kwargs.items():
            assert value == test[key]

    @pytest.mark.parametrize("confidence", (None, 0.9))
    def test_targets_from_sequence_labels(self, confidence: Optional[float]):
        # Normal case where the confidence values do not affect
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        sequence_labels = ['O', 'B', 'I', 'O', 'O', 'O', 'B', 'O', 'O']
        confidences = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        test = TargetText(text=text, text_id=text_id, sequence_labels=sequence_labels, 
                          confidence=confidences)
        test.tokenize(str.split)
        test_targets = test.get_targets_from_sequence_labels('sequence_labels', 
                                                              confidence)
        targets = ['laptop case', 'cover']
        assert targets == test_targets
        # Case where the confidence values would affect, should not include 
        # 0.9 as it has to be greater than
        confidences = [0.0, 1.0, 0.91, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]
        test['confidence'] = confidences
        test_targets = test.get_targets_from_sequence_labels('sequence_labels', 
                                                              confidence)
        if confidence is not None:
            assert ['laptop case'] == test_targets
        else:
            assert targets == test_targets
        # Case where one word in the target text spanning the B and the I's 
        # is less than the threshold and the rest are not. In this case 
        # the whole target should not be returned as the whole target 
        # has to be greater than this threshold.
        confidences = [0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0]
        test['confidence'] = confidences
        test_targets = test.get_targets_from_sequence_labels('sequence_labels', 
                                                              confidence)
        if confidence is not None:
            assert ['cover'] == test_targets
        else:
            assert targets == test_targets
        # Test the case of no targets due to confidence
        confidences = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        test['confidence'] = confidences
        test_targets = test.get_targets_from_sequence_labels('sequence_labels', 
                                                              confidence)
        if confidence is not None:
            assert [] == test_targets
        else:
            assert targets == test_targets
        # Test the case of no targets as it found none, as well as using a 
        # difference sequence key
        pred_labels = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        test['predicted_sequence_labels'] = pred_labels
        test_targets = test.get_targets_from_sequence_labels('predicted_sequence_labels', 
                                                              confidence)
        assert [] == test_targets
        # Test the case where two different targets are next to each other
        text = 'Camera screen display'
        text_id = '2'
        sequence_labels = ['B', 'B', 'I']
        confidences = [1.0, 0.9, 0.8]

        # Test the anonymised case
        test = TargetText(text=text, text_id=text_id, sequence_labels=sequence_labels, 
                          confidence=confidences, anonymised=True)
        with pytest.raises(AnonymisedError):
            test.get_targets_from_sequence_labels('predicted_sequence_labels', 
                                                  confidence)

        test = TargetText(text=text, text_id=text_id, sequence_labels=sequence_labels, 
                          confidence=confidences)
        test.tokenize(str.split)
        test_targets = test.get_targets_from_sequence_labels('sequence_labels', 
                                                              confidence)
        targets = ['Camera', 'screen display']
        if confidence is not None:
            ['Camera']
        else:
            assert targets == test_targets
        # Test the tokenized key error
        test = TargetText(text=text, text_id=text_id, sequence_labels=sequence_labels, 
                          confidence=confidences)
        with pytest.raises(KeyError):
            test.get_targets_from_sequence_labels('sequence_labels', confidence)
        # Test that the KeyError raises when no confidence key is there for 
        # the confidence case
        test = TargetText(text=text, text_id=text_id, sequence_labels=sequence_labels)
        test.tokenize(str.split)
        if confidence is not None:
            with pytest.raises(KeyError):
                test.get_targets_from_sequence_labels('sequence_labels', confidence)
        else:
            test_targets = test.get_targets_from_sequence_labels('sequence_labels', 
                                                                 confidence)
            assert targets == test_targets
        # Test the case when the confidence is not between 0 and 1
        test['confidence'] = confidences
        wrong_confidence = 1.1
        if confidence is not None:
            with pytest.raises(ValueError):
                test.get_targets_from_sequence_labels('sequence_labels', wrong_confidence)
        wrong_confidence = -0.1
        if confidence is not None:
            with pytest.raises(ValueError):
                test.get_targets_from_sequence_labels('sequence_labels', wrong_confidence)
    
    def test_targets_from_spans(self):
        # Normal case
        text = 'The laptop case was great and cover was rubbish'
        spans = [Span(4, 15), Span(30, 35)]
        test_targets = TargetText.targets_from_spans(text, spans)
        assert ['laptop case', 'cover'] == test_targets
        # Case just one target
        spans = [Span(4, 15)]
        test_targets = TargetText.targets_from_spans(text, spans)
        assert ['laptop case'] == test_targets
        # Case where the target is at the begining
        spans = [Span(0, 3)]
        test_targets = TargetText.targets_from_spans(text, spans)
        assert ['The'] == test_targets
        # Case where the target is at the end
        spans = [Span(40, 47)]
        test_targets = TargetText.targets_from_spans(text, spans)
        assert ['rubbish'] == test_targets
        # Case with no spans
        spans = []
        test_targets = TargetText.targets_from_spans(text, spans)
        assert [] == test_targets

    @pytest.mark.parametrize("confidence", (None, 0.9))
    def test_target_text_from_prediction(self, confidence: float):
        text = 'The laptop case was great and cover was rubbish'
        tokens = ['The', 'laptop', 'case', 'was', 'great', 'and', 'cover', 
                  'was', 'rubbish']
        sequence_labels = ['O', 'B', 'I', 'O', 'O', 'O', 'B', 'O', 'O']
        confidences = [0.0, 1.0, 0.91, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0]
        targets = ['laptop case', 'cover']
        spans = [Span(4, 15), Span(30, 35)]
        text_id = '1'
        # Test the normal case where the confidence and non-confidence are the 
        # same
        test= TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                     sequence_labels=sequence_labels,
                                                     tokenized_text=tokens, 
                                                     confidence=confidence, 
                                                     confidences=confidences)
        correct_dict = {'text_id': text_id, 'text': text, 'targets': targets,
                        'spans': spans, 'tokenized_text': tokens, 
                        'sequence_labels': sequence_labels, 
                        'confidence': confidences, 'categories': None,
                        'target_sentiments': None, 'category_sentiments': None}
        assert isinstance(test, TargetText)
        assert len(correct_dict) == len(test)
        for key, value in correct_dict.items():
            assert value == test[key]
        # Test the case where the confidence and non-confidence are different
        confidences = [0.0, 1.0, 0.89, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0]
        test= TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                     sequence_labels=sequence_labels,
                                                     tokenized_text=tokens, 
                                                     confidence=confidence, 
                                                     confidences=confidences)
        correct_dict['confidence'] = confidences
        if confidence is not None:
            correct_dict['spans'] = [Span(30, 35)]
            correct_dict['targets'] = ['cover']
        assert isinstance(test, TargetText)
        assert len(correct_dict) == len(test)
        for key, value in correct_dict.items():
            assert value == test[key]
        # Test the case where there are no targets or spans
        confidences = [0.5, 1.0, 0.89, 0.0, 0.0, 0.0, 0.95, 0.9, 0.0]
        sequence_labels = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        test= TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                     sequence_labels=sequence_labels,
                                                     tokenized_text=tokens, 
                                                     confidence=confidence, 
                                                     confidences=confidences)
        correct_dict['confidence'] = confidences
        correct_dict['targets'] = []
        correct_dict['spans'] = []
        correct_dict['sequence_labels'] = sequence_labels
        assert isinstance(test, TargetText)
        assert len(correct_dict) == len(test)
        for key, value in correct_dict.items():
            assert value == test[key]
        # Ensure that additional data can be given
        test= TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                     sequence_labels=sequence_labels,
                                                     tokenized_text=tokens, 
                                                     confidence=confidence, 
                                                     confidences=confidences,
                                                     pos_tags=['NN'])
        correct_dict['pos_tags'] = ['NN'] 
        assert isinstance(test, TargetText)
        assert len(correct_dict) == len(test)
        for key, value in correct_dict.items():
            assert value == test[key]
        # Test the Value Error case where the sequence labels are not the same 
        # size as the other lists
        sequence_labels = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        with pytest.raises(ValueError):
            TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                     sequence_labels=sequence_labels,
                                                     tokenized_text=tokens, 
                                                     confidence=confidence, 
                                                     confidences=confidences)
        # Test the case where the confidences are not the same length as the 
        # other lists
        confidences = [0.5, 1.0, 0.89, 0.0, 0.0, 0.0, 0.95, 0.9]
        sequence_labels = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        if confidence is not None:
            with pytest.raises(ValueError):
                TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                     sequence_labels=sequence_labels,
                                                     tokenized_text=tokens, 
                                                     confidence=confidence, 
                                                     confidences=confidences)
        else:
            TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                     sequence_labels=sequence_labels,
                                                     tokenized_text=tokens, 
                                                     confidence=confidence, 
                                                     confidences=confidences)
        # Test that a ValueError raises if any of the additional data is the
        # following; targets, spans
        confidences = [0.5, 1.0, 0.89, 0.0, 0.0, 0.0, 0.95, 0.9, 0.0]
        with pytest.raises(ValueError):
            TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                   sequence_labels=sequence_labels,
                                                   tokenized_text=tokens, 
                                                   confidence=confidence, 
                                                   confidences=confidences,
                                                   targets=[])
        with pytest.raises(ValueError):
            TargetText.target_text_from_prediction(text=text, text_id=text_id,
                                                   sequence_labels=sequence_labels,
                                                   tokenized_text=tokens, 
                                                   confidence=confidence, 
                                                   confidences=confidences,
                                                   spans=[])
    
    @pytest.mark.parametrize("incl_target", (False, True))
    def test_left_right_target_contexts(self, incl_target: bool):
        def add_target_left_right(answer: List[Tuple[List[str], List[str], List[str]]],
                                  target_object: TargetText) -> List[Tuple[List[str], List[str], List[str]]]:
            targets = target_object['targets']
            temp_answers = []
            for target_answer, target_word in zip(answer, targets):
                left, right, target = target_answer
                left += target_word
                right = f'{target_word}{right}'
                temp_answers.append((left, right, target))
            return temp_answers

        text = 'The laptop case was great and cover was rubbish'
        text_ids = ['4', '5', '6']
        spans = [[Span(0, 4)], [Span(43, 47)], None]
        targets = [['The '], ['bish'], None]
        edge_examples = []
        for i in range(3):
            example = TargetText(text, text_ids[i], targets=targets[i],
                                 spans=spans[i])
            edge_examples.append(example)
        
        normal_examples, _ = self._regular_examples()
        all_examples = normal_examples + edge_examples
        correct_answers = [[('The ', ' was great and cover was rubbish', 'laptop case')],
                           [('The laptop case was great and ', ' was rubbish', 'cover')],
                           [('The ', ' was great and cover was rubbish', 'laptop case'),
                           ('The laptop case was great and ', ' was rubbish', 'cover')],
                           [('', 'laptop case was great and cover was rubbish', 'The ')],
                           [('The laptop case was great and cover was rub', '', 'bish')],
                           []]
        if incl_target:
            temp_correct_answers = []
            for correct_answer, target_text_object in zip(correct_answers, all_examples):
                if correct_answer == []:
                    temp_correct_answers.append([])
                else:
                    temp_correct_answers.append(add_target_left_right(correct_answer, target_text_object))
            correct_answers = temp_correct_answers
        for i, example in enumerate(normal_examples):
            answer = correct_answers[i]
            test_answer = example.left_right_target_contexts(incl_target=incl_target)
            assert answer == test_answer
        # Test the anonymised case
        test = TargetText(text, text_ids[0], targets=targets[0],
                          spans=spans[0], anonymised=True)
        with pytest.raises(AnonymisedError):
            test.left_right_target_contexts(incl_target=incl_target)

    def test_target_replacement(self):
        text = 'The laptop case was great and cover was rubbish'
        targets = ['The', 'laptop case', 'great', 'cover', 'rubbish']
        spans = [Span(0,3), Span(4,15), Span(20, 25), Span(30,35), Span(40, 47)]
        sentiments = [1,0,1,0,-1]
        target_object = TargetText(text_id='1', text=text, spans=spans, 
                                   targets=targets, target_sentiments=sentiments)
        # Simple two words to one words
        new_target_object = target_object.replace_target(1, 'hello')
        assert 'The hello was great and cover was rubbish' == new_target_object['text']
        new_targets = ['The', 'hello', 'great', 'cover', 'rubbish']
        assert new_targets == new_target_object['targets']
        new_spans = [Span(0,3), Span(4,9), Span(14, 19), Span(24,29), Span(34, 41)]
        assert new_spans == new_target_object['spans']
        assert sentiments == new_target_object['target_sentiments']
        # Last target should only affect the last target
        new_target_object = target_object.replace_target(4, 'something else')
        assert 'The laptop case was great and cover was something else' == new_target_object['text']
        new_targets = ['The', 'laptop case', 'great', 'cover', 'something else']
        assert new_targets == new_target_object['targets']
        new_spans = [Span(0,3), Span(4,15), Span(20, 25), Span(30,35), Span(40, 54)]
        assert new_spans == new_target_object['spans']
        assert sentiments == new_target_object['target_sentiments']
        # The first target and thus affect all other targets
        new_target_object = target_object.replace_target(0, 'bad day')
        assert 'bad day laptop case was great and cover was rubbish' == new_target_object['text']
        assert ['bad day', 'laptop case', 'great', 'cover', 'rubbish'] == new_target_object['targets']
        new_spans = [Span(0,7), Span(8,19), Span(24, 29), Span(34,39), Span(44, 51)]
        assert new_spans == new_target_object['spans']
        assert sentiments == new_target_object['target_sentiments']
        
        # Raise a ValueError if you give an index less than 0
        with pytest.raises(ValueError):
            target_object.replace_target(-1, 'hello')
        # Raise a ValueError if you give an index greater than the number of targets
        with pytest.raises(ValueError):
            target_object.replace_target(5, 'hello')

        # Test the case where the TargetText object contains targets that 
        # overlap each other
        text = 'The laptop case was great and cover was rubbish'
        targets = ['The l', 'laptop case', 'e was great', 'cover', 'rubbish']
        spans = [Span(0,5), Span(4,15), Span(14, 25), Span(30,35), Span(40, 47)]
        sentiments = [1,0,1,0,-1]
        target_object = TargetText(text_id='1', text=text, spans=spans, 
                                   targets=targets, target_sentiments=sentiments)
        new_target_object = target_object.replace_target(3, 'bad day')
        assert 'The laptop case was great and bad day was rubbish' == new_target_object['text']
        # Should raise an error if we want to replace either 0, 1, 2
        with pytest.raises(OverLappingTargetsError):
            target_object.replace_target(0, 'hello')
        with pytest.raises(OverLappingTargetsError):
            target_object.replace_target(1, 'hello')
        with pytest.raises(OverLappingTargetsError):
            target_object.replace_target(2, 'hello')
        # Test the anonymised case
        target_object = TargetText(text_id='1', text=text, spans=spans, 
                                   targets=targets, target_sentiments=sentiments, 
                                   anonymised=True)
        with pytest.raises(AnonymisedError):
            target_object.replace_target(3, 'bad day')

    def test_in_order(self):
        for example in self._regular_examples()[0]:
            assert example.in_order()
        not_in_order = self._regular_examples()[0][-1]
        valid_spans = not_in_order._storage['spans']
        not_in_order._storage['spans'] = [valid_spans[1], valid_spans[0]]
        assert not not_in_order.in_order()

        # Tests that when two spans start with the same index but one index 
        # ends after the other it should be ordered start span and then end 
        # span if the start span is the same
        edge_case = TargetText(text_id='1', text='had a good day', 
                               targets=['good day', 'good'], 
                               spans=[Span(6,14), Span(6,10)])
        edge_case.sanitize()
        assert not edge_case.in_order()
        edge_case = TargetText(text_id='1', text='had a good day', 
                               targets=['good', 'good day'], 
                               spans=[Span(6,10), Span(6,14)])
        edge_case.sanitize()
        assert edge_case.in_order()

    def test_re_order(self):
        good_example = self._regular_examples()[0][-1]
        correct = copy.deepcopy(good_example._storage)
        good_example.re_order()
        for key, value in correct.items():
            assert value == good_example[key]
        
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        spans = [Span(30, 35), Span(4, 15)]
        target_sentiments = [1, 0]
        targets = ['cover', 'laptop case']
        categories = ['LAPTOP', 'LAPTOP#CASE']
        category_sentiments = [1, 0]
        re_order_required_example = TargetText(text=text, text_id=text_id, spans=spans, 
                                               target_sentiments=target_sentiments,
                                               targets=targets, 
                                               categories=categories, category_sentiments=category_sentiments)
        re_order_required_example.re_order()
        for key, value in correct.items():
            assert value == re_order_required_example[key]
        # ensure re-ordering twice does not make a difference
        re_order_required_example.re_order()
        for key, value in correct.items():
            assert value == re_order_required_example[key]
        # ignoring one key
        re_order_required_example = TargetText(text=text, text_id=text_id, spans=spans, 
                                               target_sentiments=target_sentiments,
                                               targets=targets, 
                                               categories=categories, category_sentiments=category_sentiments)
        re_order_required_example.re_order(keys_not_to_order=['categories'])
        for key, value in correct.items():
            if key == 'categories':
                with pytest.raises(AssertionError):
                    assert value == re_order_required_example[key]
            else:
                assert value == re_order_required_example[key]
        # that it can re-order lists that are made of list where the inner list 
        # should be sorted and not the outer list
        preds = [[1, 0], [0, 1], [1, 1], [1, 0]]
        re_order_required_example = TargetText(text=text, text_id=text_id, spans=spans, 
                                               target_sentiments=target_sentiments,
                                               targets=targets, preds=preds, 
                                               categories=categories, category_sentiments=category_sentiments)
        correct['preds'] = [[0, 1], [1, 0], [1, 1], [0, 1]]
        re_order_required_example.re_order()
        for key, value in correct.items():
            assert value == re_order_required_example[key]
        # raises IndexError when there is a key that is a list that does not 
        # have the same number of indexs as the `spans` key
        categories = ['LAPTOP']
        category_sentiments = [0]
        re_order_required_example = TargetText(text=text, text_id=text_id, spans=spans, 
                                               target_sentiments=target_sentiments,
                                               targets=targets, preds=preds, 
                                               categories=categories, category_sentiments=category_sentiments)
        with pytest.raises(Exception):
            re_order_required_example.re_order()
        # edge case where the targets is a list but with no targets in it.
        edge_case_example = TargetText(text_id=text_id, text=text, targets=[], 
                                       spans=[], target_sentiments=[])
        edge_case_example.re_order()
        assert [] == edge_case_example['targets']
        assert [] == edge_case_example['spans']
        assert [] == edge_case_example['target_sentiments']
        assert text_id == edge_case_example['text_id']
        assert text == edge_case_example['text']

        # Test the rollback case where an error occurs halfway through ordering
        spans = [Span(30, 35), Span(4, 15)]
        targets = ['cover', 'laptop case']
        rollback_case_example = TargetText(text_id=text_id, targets=targets, spans=spans,
                                           zcategories=['LAPTOP'],
                                           text=text)
        with pytest.raises(Exception):
            rollback_case_example.re_order()
        assert rollback_case_example['targets'][0] == 'cover'
        assert rollback_case_example['spans'][0] == Span(30, 35)


    @pytest.mark.parametrize("id_delimiter", ('$$', None))
    def test_add_unique_key(self, id_delimiter: Optional[str]):
        # Normal case, tested across different keys and id keys.
        examples = self._regular_examples()[0]
        text_ids = ['0', 'another_id', '2']
        for key_name, id_key_name in [('spans', 'spans_id'), 
                                      ('targets', 'targets_id')]:
            for index, example in enumerate(examples):
                example: TargetText
                assert id_key_name not in example
                if id_delimiter is not None:
                    example.add_unique_key(key_name, id_key_name, id_delimiter=id_delimiter)
                else:
                    example.add_unique_key(key_name, id_key_name)
                assert id_key_name in example
                relevant_text_id = text_ids[index]
                if id_delimiter is not None:
                    relevant_text_id = relevant_text_id + id_delimiter
                else:
                    relevant_text_id = relevant_text_id + '::'
                
                correct_ids = []
                for key_index in range(len(example[key_name])):
                    correct_ids.append(f'{relevant_text_id}{key_index}')
                assert correct_ids == example[id_key_name]
        # Test the case where the id_key_name already exists
        with pytest.raises(KeyError):
            examples[0].add_unique_key('spans', 'spans_id')
        # Test the case where the key_name is not of type List
        with pytest.raises(TypeError):
            examples[0].add_unique_key('text', 'another_id')
            
            
