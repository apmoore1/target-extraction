from typing import List
from pathlib import Path
import tempfile

import pytest

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.data_types_util import Span
from target_extraction.tokenizers import spacy_tokenizer
from target_extraction.pos_taggers import spacy_tagger

class TestTargetTextCollection:

    def _json_data_dir(self) -> Path:
        return Path(__file__, '..', 'data').resolve()

    def _target_text_measure_examples(self) -> List[TargetText]:
        text = 'The laptop case was great and cover was rubbish'
        text_id = '0'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        target_text_0 = TargetText(text=text, text_id=text_id, spans=spans, 
                                   targets=targets)
        text = 'The laptop price was awful'
        text_id = '1'
        spans = [Span(4, 16)]
        targets = ['laptop price']
        target_text_1 = TargetText(text=text, text_id=text_id, spans=spans,
                                   targets=targets)
        
        return [target_text_0, target_text_1]

    def _target_text_examples(self) -> List[TargetText]:
        text = 'The laptop case was great and cover was rubbish'
        text_ids = ['0', 'another_id', '2']
        spans = [[Span(4, 15)], [Span(30, 35)], [Span(4, 15), Span(30, 35)]]
        target_sentiments = [[0], [1], [0, 1]]
        targets = [['laptop case'], ['cover'], ['laptop case', 'cover']]
        categories = [['LAPTOP#CASE'], ['LAPTOP'], ['LAPTOP#CASE', 'LAPTOP']]

        target_text_examples = []
        for i in range(3):
            example = TargetText(text, text_ids[i], targets=targets[i],
                                 spans=spans[i], 
                                 target_sentiments=target_sentiments[i],
                                 categories=categories[i])
            target_text_examples.append(example)
        return target_text_examples

    def _target_text_not_align_example(self) -> TargetText:
        text = 'The laptop case; was awful'
        text_id = 'inf'
        spans = [Span(4, 15)]
        targets = ['laptop case']
        return TargetText(text=text, text_id=text_id, spans=spans, targets=targets)


    def _target_text_example(self) -> TargetText:
        return self._target_text_examples()[-1]


    def _regular_examples(self) -> List[TargetTextCollection]:

        target_text_examples = self._target_text_examples()
        
        collection_examples = []
        for i in range(1, 4):
            collection = TargetTextCollection()
            for example in target_text_examples[:i]:
                collection[example['text_id']] = example
            collection_examples.append(collection)
        return collection_examples

    def test_length(self):
        examples = self._regular_examples()
        example_lengths = [1,2,3]
        for example, real_length in zip(examples, example_lengths):
            assert len(example) == real_length

    def test_eq(self):
        examples = self._regular_examples()

        example_0 = examples[0]
        example_1 = examples[1]
        example_2 = examples[2]
        assert example_0 != example_1

        assert example_2 == example_2
        assert example_0 == example_0

        del example_2['2']
        assert example_2 == example_1

        # Ensure that it only relies on the text_id and not the content of the 
        # target text
        example_2['2'] = TargetText('hello how', '2')
        example_1['2'] = TargetText('another test', '2')
        assert example_2 == example_1

    def test_get_item(self):
        examples = self._regular_examples()
        example_2 = examples[2]

        last_target_text = self._target_text_example()
        assert example_2['2'] == last_target_text

        assert example_2['0'] == TargetText('can be any text as long as id is correct', 
                                            '0')
        
        with pytest.raises(KeyError):
            example_2['any key']

    def test_del_item(self):
        examples = self._regular_examples()
        example_2 = examples[2]

        assert len(example_2) == 3
        del example_2['2']
        assert len(example_2) == 2

        with pytest.raises(KeyError):
            example_2['2']

    def test_set_item(self):
        new_collection = TargetTextCollection()

        # Full target text.
        new_collection['2'] = self._target_text_example()
        # Minimum target text
        new_collection['2'] = TargetText('minimum example', '2')
        example_args = {'text': 'minimum example', 'text_id': '2'}
        new_collection['2'] = TargetText(**example_args)

        with pytest.raises(ValueError):
            new_collection['2'] = TargetText('minimum example', '3')
        with pytest.raises(TypeError):
            new_collection['2'] = example_args

        # Ensure that if the given TargetText changes it does not change in 
        # the collection
        example_instance = TargetText(**example_args)
        example_collection = TargetTextCollection()
        example_collection['2'] = example_instance

        example_instance['target_sentiments'] = [0]
        assert example_instance['target_sentiments'] is not None
        assert example_collection['2']['target_sentiments'] is None

    def test_add(self):
        new_collection = TargetTextCollection()

        assert len(new_collection) == 0
        new_collection.add(self._target_text_example())
        assert len(new_collection) == 1

        assert '2' in new_collection

    def test_construction(self):
        new_collection = TargetTextCollection(name='new_name')
        assert new_collection.name == 'new_name'

        new_collection = TargetTextCollection()
        assert new_collection.name is ''

        new_collection = TargetTextCollection(target_texts=self._target_text_examples())
        assert len(new_collection) == 3
        assert '2' in new_collection

    def test_to_json(self):
        # no target text instances in the collection (empty collection)
        new_collection = TargetTextCollection()
        assert new_collection.to_json() == ''

        # One target text in the collection
        new_collection = TargetTextCollection([self._target_text_example()])
        true_json_version = ('{"text": "The laptop case was great and cover '
                             'was rubbish", "text_id": "2", "targets": ["laptop '
                             'case", "cover"], "spans": [[4, 15], [30, 35]], '
                             '"target_sentiments": [0, 1], "categories": '
                             '["LAPTOP#CASE", "LAPTOP"], "category_sentiments": null}')
        assert new_collection.to_json() == true_json_version

        # Multiple target text in the collection
        new_collection = TargetTextCollection(self._target_text_examples()[:2])
        true_json_version = ('{"text": "The laptop case was great and cover '
                             'was rubbish", "text_id": "0", "targets": '
                             '["laptop case"], "spans": [[4, 15]], '
                             '"target_sentiments": [0], "categories": '
                             '["LAPTOP#CASE"], "category_sentiments": null}'
                             '\n{"text": "The laptop case was '
                             'great and cover was rubbish", "text_id": '
                             '"another_id", "targets": ["cover"], "spans": '
                             '[[30, 35]], "target_sentiments": [1], "categories": '
                             '["LAPTOP"], "category_sentiments": null}')
        assert new_collection.to_json() == true_json_version

    @pytest.mark.parametrize("name", ('', 'test_name'))
    def test_from_json(self, name):
        # no text given
        test_collection = TargetTextCollection.from_json('', name=name)
        assert TargetTextCollection() == test_collection
        assert test_collection.name == name

        # One target text instance in the text
        new_collection = TargetTextCollection([self._target_text_example()], 
                                              name=name)
        json_one_collection = new_collection.to_json()
        assert new_collection == TargetTextCollection.from_json(json_one_collection)
        assert new_collection.name == name

        # Multiple target text instances in the text
        new_collection = TargetTextCollection(self._target_text_examples()[:2])
        json_multi_collection = new_collection.to_json()
        assert new_collection == TargetTextCollection.from_json(json_multi_collection)

    @pytest.mark.parametrize("name", ('', 'test_name'))
    def test_load_json(self, name):
        empty_json_fp = Path(self._json_data_dir(), 'empty_target_instance.json')
        empty_collection = TargetTextCollection.load_json(empty_json_fp, name=name)
        assert TargetTextCollection() == empty_collection
        assert empty_collection.name == name

        # Ensure that it raises an error when loading a bad json file
        wrong_json_fp = Path(self._json_data_dir(),
                             'wrong_target_instance.json')
        with pytest.raises(ValueError):
            TargetTextCollection.load_json(wrong_json_fp, name=name)
        # Ensure that it can load a single target text instance correctly
        one_target_json_fp = Path(self._json_data_dir(), 
                                  'one_target_instance.json')
        one_target_collection = TargetTextCollection.load_json(one_target_json_fp)
        assert len(one_target_collection) == 1
        assert one_target_collection['0']['text'] == 'The laptop case was great and cover was rubbish'
        assert one_target_collection['0']['target_sentiments'] == [0]
        assert one_target_collection['0']['category_sentiments'] == ['pos']
        assert one_target_collection['0']['categories'] == ['LAPTOP#CASE']
        assert one_target_collection['0']['spans'] == [Span(4, 15)]
        assert one_target_collection['0']['targets'] == ['laptop case']

        # Ensure that it can load multiple target text instances
        two_target_json_fp = Path(self._json_data_dir(), 'one_target_one_empty_instance.json')
        two_target_collection = TargetTextCollection.load_json(two_target_json_fp)
        assert len(two_target_collection) == 2

    def test_to_json_file(self):
        test_collection = TargetTextCollection()
        with tempfile.NamedTemporaryFile(mode='w+') as temp_fp:
            temp_path = Path(temp_fp.name)
            test_collection.to_json_file(temp_path)
            assert len(TargetTextCollection.load_json(temp_path)) == 0

            # Ensure that it can load more than one Target Text examples
            test_collection = TargetTextCollection(self._target_text_examples())
            test_collection.to_json_file(temp_path)
            assert len(TargetTextCollection.load_json(temp_path)) == 3

            # Ensure that if it saves to the same file that it overwrites that 
            # file
            test_collection = TargetTextCollection(self._target_text_examples())
            test_collection.to_json_file(temp_path)
            assert len(TargetTextCollection.load_json(temp_path)) == 3

            # Ensure that it can just load one examples
            test_collection = TargetTextCollection([self._target_text_example()])
            test_collection.to_json_file(temp_path)
            assert len(TargetTextCollection.load_json(temp_path)) == 1
    
    def test_tokenize(self):
        # Test the normal case with one TargetText Instance in the collection
        test_collection = TargetTextCollection([self._target_text_example()])
        test_collection.tokenize(str.split)
        tokenized_answer = ['The', 'laptop', 'case', 'was', 'great', 'and', 
                            'cover', 'was', 'rubbish']
        test_collection['2']['tokenized_text'] = tokenized_answer

        # Test the normal case with multiple TargetText Instance in the 
        # collection
        test_collection = TargetTextCollection(self._target_text_examples())
        test_collection.tokenize(spacy_tokenizer())
        test_collection['2']['tokenized_text'] = tokenized_answer

    def test_pos_text(self):
        # Test the normal case with one TargetText Instance in the collection
        test_collection = TargetTextCollection([self._target_text_example()])
        test_collection.tokenize(spacy_tokenizer())
        test_collection.pos_text(spacy_tagger())
        pos_answer = ['DET', 'NOUN', 'NOUN', 'VERB', 'ADJ', 'CCONJ', 'NOUN', 
                      'VERB', 'ADJ']
        assert test_collection['2']['pos_tags'] == pos_answer

        # Test the normal case with multiple TargetText Instance in the 
        # collection
        test_collection = TargetTextCollection(self._target_text_examples())
        test_collection.tokenize(spacy_tokenizer())
        test_collection.pos_text(spacy_tagger())
        assert test_collection['2']['pos_tags'] == pos_answer

        # Test the case where the tagger function given does not return a 
        # List
        with pytest.raises(TypeError):
            test_collection.pos_text(str.strip)
        # Test the case where the tagger function given returns a list but 
        # not a list of strings
        token_len = lambda text: [len(token) for token in text.split()]
        with pytest.raises(TypeError):
            test_collection.pos_text(token_len)
        # Test the case where the TargetTextCollection has not be tokenized
        test_collection = TargetTextCollection([self._target_text_example()])
        with pytest.raises(ValueError):
            test_collection.pos_text(spacy_tagger())
        # Test the case where the tokenization is different to the POS tagger
        text = 'Hello how are you? I am good thank you'
        target_text_example = TargetText(text=text, text_id='1')
        test_collection = TargetTextCollection([target_text_example])
        test_collection.tokenize(str.split)
        with pytest.raises(ValueError):
            test_collection.pos_text(spacy_tagger())

    def test_force_targets(self):
        text = 'The laptop casewas great and cover was rubbish'
        spans = [Span(4, 15), Span(29, 34)]
        targets = ['laptop case', 'cover']
        target_text = TargetText(text=text, text_id='0', targets=targets, 
                                 spans=spans)
        text_1 = 'The laptop casewas great andcover was rubbish'
        spans_1 = [Span(4, 15), Span(28, 33)]
        target_text_1 = TargetText(text=text_1, text_id='1', targets=targets,
                                   spans=spans_1)

        perfect_text = 'The laptop case was great and cover was rubbish'
        perfect_spans = [Span(4, 15), Span(30, 35)]

        # Test the single case
        test_collection = TargetTextCollection([target_text])
        test_collection.force_targets()
        assert test_collection['0']['text'] == perfect_text
        assert test_collection['0']['spans'] == perfect_spans

        # Test the multiple case
        test_collection = TargetTextCollection([target_text, target_text_1])
        test_collection.force_targets()
        for target_key in ['0', '1']:
            assert test_collection[target_key]['text'] == perfect_text
            assert test_collection[target_key]['spans'] == perfect_spans

    def test_sequence_labels(self):
        # Test the single case
        test_collection = TargetTextCollection([self._target_text_example()])
        test_collection.tokenize(spacy_tokenizer())
        test_collection.sequence_labels()
        correct_sequence = ['O', 'B', 'I', 'O', 'O', 'O', 'B', 'O', 'O']
        assert test_collection['2']['sequence_labels'] == correct_sequence

        # Test the multiple case
        test_collection = TargetTextCollection(self._target_text_examples())
        test_collection.tokenize(spacy_tokenizer())
        test_collection.sequence_labels()
        correct_sequence = ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O']
        assert test_collection['another_id']['sequence_labels'] == correct_sequence

    def test_exact_match_score(self):
        # Simple case where it should get perfect score
        test_collection = TargetTextCollection([self._target_text_example()])
        test_collection.tokenize(spacy_tokenizer())
        test_collection.sequence_labels()
        measures = test_collection.exact_match_score('sequence_labels')
        for measure in measures:
            assert measure == 1.0

        # Something that has perfect precision but misses one therefore does 
        # not have perfect recall nor f1
        test_collection = TargetTextCollection(self._target_text_measure_examples())
        test_collection.tokenize(str.split)
        # text = 'The laptop case was great and cover was rubbish'
        sequence_labels_0 = ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O']
        test_collection['0']['sequence_labels'] = sequence_labels_0
        # text = 'The laptop price was awful'
        sequence_labels_1 = ['O', 'B', 'I', 'O', 'O']
        test_collection['1']['sequence_labels'] = sequence_labels_1
        recall, precision, f1 = test_collection.exact_match_score('sequence_labels')
        assert precision == 1.0
        assert recall == 2.0/3.0
        assert f1 == 0.8

        # Something that has perfect recall but not precision as it over 
        # predicts 
        sequence_labels_0 = ['O', 'B', 'I', 'B', 'O', 'O', 'B', 'O', 'O']
        test_collection['0']['sequence_labels'] = sequence_labels_0
        sequence_labels_1 = ['O', 'B', 'I', 'O', 'O']
        test_collection['1']['sequence_labels'] = sequence_labels_1
        recall, precision, f1 = test_collection.exact_match_score(
            'sequence_labels')
        assert precision == 3/4
        assert recall == 1.0
        assert round(f1, 3) == 0.857

        # Does not predict anything for a whole sentence therefore will have 
        # perfect precision but bad recall (mainly testing the if not 
        # getting anything for a sentence matters)
        sequence_labels_0 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        test_collection['0']['sequence_labels'] = sequence_labels_0
        sequence_labels_1 = ['O', 'B', 'I', 'O', 'O']
        test_collection['1']['sequence_labels'] = sequence_labels_1
        recall, precision, f1 = test_collection.exact_match_score(
            'sequence_labels')
        assert precision == 1.0
        assert recall == 1/3
        assert f1 == 0.5

        # Handle the edge case of not getting anything
        sequence_labels_0 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        test_collection['0']['sequence_labels'] = sequence_labels_0
        sequence_labels_1 = ['O', 'O', 'O', 'O', 'O']
        test_collection['1']['sequence_labels'] = sequence_labels_1
        recall, precision, f1 = test_collection.exact_match_score('sequence_labels')
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

        # The case where the tokens and the text do not align
        not_align_example = self._target_text_not_align_example()
        # text = 'The laptop case; was awful'
        sequence_labels_align = ['O', 'B', 'I', 'O', 'O']
        test_collection.add(not_align_example)
        test_collection.tokenize(str.split)
        test_collection['inf']['sequence_labels'] = sequence_labels_align
        sequence_labels_0 = ['O', 'B', 'I', 'O', 'O', 'O', 'B', 'O', 'O']
        test_collection['0']['sequence_labels'] = sequence_labels_0
        sequence_labels_1 = ['O', 'B', 'I', 'O', 'O']
        test_collection['1']['sequence_labels'] = sequence_labels_1
        recall, precision, f1 = test_collection.exact_match_score('sequence_labels')
        assert recall == 3/4
        assert precision == 3/4
        assert f1 == 0.75

        # This time it can get a perfect score as the token alignment will be 
        # perfect
        test_collection.tokenize(spacy_tokenizer())
        sequence_labels_align = ['O', 'B', 'I', 'O', 'O', 'O']
        test_collection['inf']['sequence_labels'] = sequence_labels_align
        recall, precision, f1 = test_collection.exact_match_score('sequence_labels')
        assert recall == 1.0
        assert precision == 1.0
        assert f1 == 1.0

        # Handle the case where one of the samples has no spans
        test_example = TargetText(text="I've had a bad day", text_id='50')
        other_examples = self._target_text_measure_examples()
        other_examples.append(test_example)
        test_collection = TargetTextCollection(other_examples)
        test_collection.tokenize(str.split)
        test_collection.sequence_labels()
        measures = test_collection.exact_match_score('sequence_labels')
        for measure in measures:
            assert measure == 1.0
        # Handle the case where on the samples has no spans but has predicted 
        # there is a span there
        test_collection['50']['sequence_labels'] = ['B', 'I', 'O', 'O', 'O']
        recall, precision, f1 = test_collection.exact_match_score('sequence_labels')
        assert recall == 1.0
        assert precision == 3/4
        assert round(f1, 3) == 0.857


    def test_samples_with_targets(self):
        # Test the case where all of the TargetTextCollection contain targets
        test_collection = TargetTextCollection(self._target_text_examples())
        sub_collection = test_collection.samples_with_targets()
        assert test_collection == sub_collection
        assert len(test_collection) == 3

        # Test the case where none of the TargetTextCollection contain targets
        for sample_id in list(test_collection.keys()):
            del test_collection[sample_id]
            test_collection.add(TargetText(text='nothing', text_id=sample_id))
        print(test_collection)
        assert len(test_collection) == 3
        sub_collection = test_collection.samples_with_targets()
        assert len(sub_collection) == 0
        assert sub_collection != test_collection

        # Test the case where only a 2 of the the three TargetTextCollection 
        # contain targets.
        test_collection = TargetTextCollection(self._target_text_examples())
        del test_collection['another_id']
        easy_case = TargetText(text='something else', text_id='another_id')
        test_collection.add(easy_case)
        sub_collection = test_collection.samples_with_targets()
        assert len(sub_collection) == 2
        assert sub_collection != test_collection

        # Test the case where the targets are just an empty list rather than 
        # None
        test_collection = TargetTextCollection(self._target_text_examples())
        del test_collection['another_id']
        edge_case = TargetText(text='something else', text_id='another_id', 
                               targets=[], spans=[])
        test_collection.add(edge_case)
        sub_collection = test_collection.samples_with_targets()
        assert len(sub_collection) == 2
        assert sub_collection != test_collection

    def test_target_count(self):
        # Start with an empty collection
        test_collection = TargetTextCollection()
        nothing = test_collection.target_count()
        assert len(nothing) == 0
        assert not nothing

        # Collection that contains TargetText instances but with no targets
        test_collection.add(TargetText(text='some text', text_id='1'))
        assert len(test_collection) == 1
        nothing = test_collection.target_count()
        assert len(nothing) == 0
        assert not nothing

        # Collection now contains at least one target
        test_collection.add(TargetText(text='another item today', text_id='2',
                                       spans=[Span(0, 12)], 
                                       targets=['another item']))
        assert len(test_collection) == 2
        one = test_collection.target_count()
        assert len(one) == 1
        assert one == {'another item': 1}

        # Collection now contains 3 targets but 2 are the same
        test_collection.add(TargetText(text='another item today', text_id='3',
                                       spans=[Span(0, 12)], 
                                       targets=['another item']))
        test_collection.add(TargetText(text='item today', text_id='4',
                                       spans=[Span(0, 4)], 
                                       targets=['item']))
        assert len(test_collection) == 4
        two = test_collection.target_count()
        assert len(two) == 2
        assert two == {'another item': 2, 'item': 1}
    
    @pytest.mark.parametrize("remove_empty", (False, True))
    def test_one_sample_per_span(self, remove_empty: bool):
        # Case where nothing should change with respect to the number of spans 
        # but will change the values target_sentiments to None etc
        target_text = TargetText(text_id='0', spans=[Span(4, 15)], 
                                 text='The laptop case was great and cover was rubbish',
                                 target_sentiments=[0], targets=['laptop case'])
        collection = TargetTextCollection([target_text])
        new_collection = collection.one_sample_per_span(remove_empty=remove_empty)
        assert new_collection == collection
        assert new_collection['0']['spans'] == [Span(4, 15)]
        assert new_collection['0']['target_sentiments'] == None
        assert collection['0']['target_sentiments'] == [0]

        # Should change the number of Spans.
        assert target_text['target_sentiments'] == [0]
        target_text._storage['spans'] = [Span(4, 15), Span(4, 15)]
        target_text._storage['targets'] = ['laptop case', 'laptop case']
        target_text._storage['target_sentiments'] = [0,1]
        diff_collection = TargetTextCollection([target_text])
        new_collection = diff_collection.one_sample_per_span(remove_empty=remove_empty)
        assert new_collection == collection
        assert new_collection['0']['spans'] == [Span(4, 15)]
        assert new_collection['0']['target_sentiments'] == None
        assert diff_collection['0']['target_sentiments'] == [0, 1]
        assert diff_collection['0']['spans'] == [Span(4, 15), Span(4, 15)]

    def test_sanitize(self):
        # The normal case where no errors should be raised.
        target_text = TargetText(text_id='0', spans=[Span(4, 15)], 
                                 text='The laptop case was great and cover was rubbish',
                                 target_sentiments=[0], targets=['laptop case'])
        collection = TargetTextCollection([target_text])
        collection.sanitize()

        # The case where an error should be raised
        with pytest.raises(ValueError):
            target_text._storage['spans'] = [Span(3,15)]
            collection = TargetTextCollection([target_text])
            collection.sanitize()









        
        


