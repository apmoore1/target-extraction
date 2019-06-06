import copy
from itertools import repeat
import traceback
from typing import List, Dict, Any, Tuple, Iterable, Callable

import pytest

from target_extraction.data_types import TargetText, Span
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

    def _exception_examples(self) -> Iterable[Dict[str, Any]]:
        '''
        :returns: The opposite of _passable_examples this returns a list of
                  key word arguments to give to the constructor of 
                  TargetText that SHOULD raise a ValueError by the 
                  check_list_sizes function.
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
            yield {'text_id': text_ids[0], 'text': texts[0],
                   'targets': target, 'spans': span, 
                   'target_sentiments': target_sentiment}
        # Category and Category sentiments mismatch
        category_category_sentiments = [[possible_categories[0], possible_category_sentiments[2]],
                                        [possible_categories[2], possible_category_sentiments[0]]]
        for categories, category_sentiment in category_category_sentiments:
            yield {'text_id': text_ids[0], 'text': texts[0],
                   'category_sentiments': category_sentiment,
                   'categories': categories}

        # Shouldn't work as the target does not have a reference span
        values = zip(text_ids, texts, possible_targets,
                     possibe_target_sentiments, possible_categories,
                     possible_category_sentiments)
        for _id, text, target, target_sentiment, category, category_sentiment in values:
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
        values = zip(text_ids, texts, span_target_mismatchs)
        for _id, text, span_target in values:
            span, target = span_target
            yield {'text_id': _id, 'text': text, 'spans': span,
                   'targets': target}

    def _passable_examples(self) -> List[Tuple[str, Dict[str, Any]]]:
        '''
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

    def test_check_list_sizes(self):
        for passable_err_msg, passable_arguments in self._passable_examples():
            try:
                TargetText(**passable_arguments)
            except:
                traceback.print_exc()
                raise Exception(passable_err_msg)
        for value_error_arguments in self._exception_examples():
            with pytest.raises(ValueError):
                TargetText(**value_error_arguments)

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
        key and that it follows the check_list_sizes sanity checks.
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

    def test_from_json(self):
        json_text = ('{"text": "The laptop case was great and cover was rubbish", '
                     '"text_id": "2", "targets": ["laptop case", "cover"], '
                     '"spans": [[4, 15], [30, 35]], "target_sentiments": [0, 1], '
                     '"categories": ["LAPTOP#CASE", "LAPTOP"], "category_sentiments": [0, 1]}')
        example_from_json = TargetText.from_json(json_text)
        example_spans: List[Span] = example_from_json['spans']
        for span in example_spans:
            assert isinstance(span, Span), f'{span} should be of type Span'
        
        examples, _ = self._regular_examples()
        example = examples[-1]
        for key, value in example.items():
            assert value == example_from_json[key]

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
        test_target_text['tokenized_text'] = tokenized_answer

        # Test the case where the tokenizer function given does not return a
        # List
        test_target_text = TargetText(text=text, text_id=text_id)
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.tokenize(str.strip, perform_type_checks=type_checks)
        else:
            # Raises an error through _is_character_preserving
            with pytest.raises(ValueError):
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
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        tokenizer = spacy_tokenizer()
        pos_tagger = spacy_tagger()
        test_target_text = TargetText(text=text, text_id=text_id)
        test_target_text.tokenize(tokenizer)
        test_target_text.pos_text(pos_tagger, perform_type_checks=type_checks)

        pos_answer = ['DET', 'NOUN', 'NOUN', 'VERB', 'ADJ', 'CCONJ', 'NOUN',
                      'VERB', 'ADJ']
        test_target_text['pos_tags'] = pos_answer

        # Test the case where the tagger function given does not return a
        # List
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.pos_text(str.strip, perform_type_checks=type_checks)
        else:
            with pytest.raises(ValueError):
                test_target_text.pos_text(str.strip, perform_type_checks=type_checks)
        
        # Test the case where the tagger function given returns a list but
        # not a list of strings
        def token_len(text): return [len(token) for token in text.split()]
        if type_checks:
            with pytest.raises(TypeError):
                test_target_text.pos_text(token_len, perform_type_checks=type_checks)
        else:
            test_target_text.pos_text(token_len, perform_type_checks=type_checks)
        del test_target_text['tokenized_text']
        # Test the case where the TargetText has not be tokenized
        with pytest.raises(ValueError):
            test_target_text.pos_text(pos_tagger)
        # Test the case where the tokenization is different to the POS tagger
        text = 'Hello how are you? I am good thank you'
        target_text_example = TargetText(text=text, text_id='1')
        target_text_example.tokenize(str.split)
        with pytest.raises(ValueError):
            target_text_example.pos_text(pos_tagger)

    def test_sequence_labels(self):
        # Test that it will raise a KeyError if it has not been tokenized
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        with pytest.raises(KeyError):
            test.sequence_labels()
        # Test that it will raise a KeyError if it has been tokenized but no 
        # targets or spans
        test = TargetText(text=text, text_id=text_id)
        test.tokenize(str.split)
        with pytest.raises(KeyError):
            test.sequence_labels()
        # Test the basic case where we have only one target of one word
        text = 'The laptop'
        spans = [Span(4, 10)]
        targets = ['laptop']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['O', 'B']
        # Test the case where the sequence is greater more than one word
        text = 'The laptop case'
        spans = [Span(4, 15)]
        targets = ['laptop case']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['O', 'B', 'I']
        # Test the case where the target is the first word
        text = 'laptop case is great'
        spans = [Span(0, 11)]
        targets = ['laptop case']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['B', 'I', 'O', 'O']
        # Test the case where there is more than one target
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['O', 'B', 'I', 'O', 'O', 'O', 
                                           'B', 'O', 'O']
        # Test the case where the tokens do not perfectly align the target text 
        # spans and therefore only match on one of the perfectly aligned tokens
        text = 'The laptop;priced was high and the laptop cover-ed was based'
        text_id = '2'
        spans = [Span(11, 17), Span(35, 47)]
        targets = ['priced', 'laptop cover']
        test = TargetText(text=text, text_id=text_id, spans=spans,
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['O', 'O', 'O', 'O', 'O', 'O',
                                           'B', 'O', 'O', 'O']
        # Different tokenizer would get a different result
        test.tokenize(spacy_tokenizer())
        test.sequence_labels()
        assert test['sequence_labels'] == ['O', 'O', 'O', 'O', 'O', 'O',
                                           'B', 'I', 'O', 'O', 'O', 'O']
        test.tokenize(stanford())
        test.sequence_labels()
        ['The', 'laptop', ';', 'priced', 'was', 'high', 'and',
            'the', 'laptop', 'cover', '-', 'ed', 'was', 'based']
        assert test['sequence_labels'] == ['O', 'O', 'O', 'B', 'O', 'O',
                                           'O', 'O', 'B', 'I', 'O', 'O', 'O', 'O']
        # Test if using force targets will correct the above mistakes which it
        # should
        test.force_targets()
        assert test['text'] == 'The laptop; priced was high and the laptop cover -ed was based'
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['O', 'O', 'B', 'O', 'O', 'O', 'O',
                                           'B', 'I', 'O', 'O', 'O']
        
        # Case where two targets are next to each other.
        text = 'The laptop case price was great and bad'
        spans = [Span(4, 15), Span(16, 21)]
        targets = ['laptop case', 'price']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['O', 'B', 'I', 'B', 'O', 'O', 'O',
                                           'O']

        # Case where the targets are at the start and end and contain no `O`
        # Handle cases at the extreme of the sentence (beginning and end)
        text = 'Laptop priced'
        spans = [Span(0, 6), Span(7, 13)]
        targets = ['Laptop', 'priced']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['B', 'B']
        
        # Handle the case where the targets are next to each other where one 
        # ends with an I tag and the other starts with a B tag.
        text = 'Laptop cover priced'
        spans = [Span(0, 12), Span(13, 19)]
        targets = ['Laptop cover', 'priced']
        test = TargetText(text=text, text_id=text_id, spans=spans, 
                          targets=targets)
        test.tokenize(str.split)
        test.sequence_labels()
        assert test['sequence_labels'] == ['B', 'I', 'B']

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

    def test_get_sequence_spans(self):
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