from typing import List, Optional, Dict, Any
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

    def test_dict_iterator(self):
        collections = self._regular_examples()[:2]

        target_0 = {'text': 'The laptop case was great and cover was rubbish', 
                    'text_id': '0', 'spans': [Span(4,15)], 
                    'targets': ['laptop case'], 
                    'categories': ['LAPTOP#CASE'], 'target_sentiments': [0],
                    'category_sentiments': None}
        target_1 = {'text': 'The laptop case was great and cover was rubbish', 
                    'text_id': 'another_id', 'spans': [Span(30,35)], 
                    'targets': ['cover'], 'category_sentiments': None,
                    'categories': ['LAPTOP'], 'target_sentiments': [1]}
        answers = [[target_0], [target_0, target_1]]
        for index, collection in enumerate(collections):
            answer = answers[index]
            for answer_index, target_dict in enumerate(collection.dict_iterator()):
                assert answer[answer_index] == target_dict


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

        # Testing instance, if not TargetTextCollection should return False
        text = 'today was good'
        text_id = '1'
        dict_version = [{'text_id': text_id, 'text': text}]
        collection_version = TargetTextCollection([TargetText(**dict_version[0])])
        assert dict_version != collection_version

        # Should return False as they have different text_id's but have the 
        # content
        alt_collection_version = TargetTextCollection([TargetText(text_id='2',
                                                                  text=text)])
        assert collection_version != alt_collection_version
        assert len(collection_version) == len(alt_collection_version)


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
        assert new_collection.metadata == {'name': 'new_name'}

        new_collection = TargetTextCollection()
        assert new_collection.name == ''
        assert new_collection.metadata is None

        example_metadata = {'model': 'InterAE'}
        new_collection = TargetTextCollection(metadata=example_metadata)
        assert new_collection.name == ''
        assert new_collection.metadata == example_metadata

        example_metadata = {'model': 'InterAE'}
        new_collection = TargetTextCollection(metadata=example_metadata,
                                              name='new_name')
        assert new_collection.name == 'new_name'
        assert new_collection.metadata == {**example_metadata, 'name': 'new_name'}

        new_collection = TargetTextCollection(target_texts=self._target_text_examples())
        assert len(new_collection) == 3
        assert '2' in new_collection
        assert new_collection.metadata is None
        assert new_collection.name == ''

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

        # Test that adding the metadata works
        new_collection = TargetTextCollection(self._target_text_examples()[:2], 
                                              metadata={'model': 'InterAE'})
        true_json_version = ('{"text": "The laptop case was great and cover '
                             'was rubbish", "text_id": "0", "targets": '
                             '["laptop case"], "spans": [[4, 15]], '
                             '"target_sentiments": [0], "categories": '
                             '["LAPTOP#CASE"], "category_sentiments": null}'
                             '\n{"text": "The laptop case was '
                             'great and cover was rubbish", "text_id": '
                             '"another_id", "targets": ["cover"], "spans": '
                             '[[30, 35]], "target_sentiments": [1], "categories": '
                             '["LAPTOP"], "category_sentiments": null}\n'
                             '{"metadata": {"model": "InterAE"}}')
        assert new_collection.to_json() == true_json_version
        
        # Test that adding the name works
        new_collection = TargetTextCollection(self._target_text_examples()[:2], 
                                              name='test name')
        true_json_version = ('{"text": "The laptop case was great and cover '
                             'was rubbish", "text_id": "0", "targets": '
                             '["laptop case"], "spans": [[4, 15]], '
                             '"target_sentiments": [0], "categories": '
                             '["LAPTOP#CASE"], "category_sentiments": null}'
                             '\n{"text": "The laptop case was '
                             'great and cover was rubbish", "text_id": '
                             '"another_id", "targets": ["cover"], "spans": '
                             '[[30, 35]], "target_sentiments": [1], "categories": '
                             '["LAPTOP"], "category_sentiments": null}\n'
                             '{"metadata": {"name": "test name"}}')
        assert new_collection.to_json() == true_json_version

    @pytest.mark.parametrize("name", (' ', 'test_name'))
    @pytest.mark.parametrize("metadata", (None, {'model': 'InterAE'}))
    def test_from_json(self, name: str, metadata: Optional[Dict[str, Any]]):
        if metadata is None:
            metadata = {}
            metadata['name'] = name
        # no text given
        test_collection = TargetTextCollection.from_json('', name=name, 
                                                         metadata=metadata)
        assert TargetTextCollection() == test_collection
        assert test_collection.name == name
        assert test_collection.metadata == metadata

        # One target text instance in the text
        new_collection = TargetTextCollection([self._target_text_example()], 
                                              name=name, metadata=metadata)
        json_one_collection = new_collection.to_json()
        assert new_collection == TargetTextCollection.from_json(json_one_collection)
        assert new_collection.name == name
        assert new_collection.metadata == metadata

        # Multiple target text instances in the text
        new_collection = TargetTextCollection(self._target_text_examples()[:2], 
                                              name=name, metadata=metadata)
        json_multi_collection = new_collection.to_json()
        assert new_collection == TargetTextCollection.from_json(json_multi_collection)
        assert new_collection.name == name
        assert new_collection.metadata == metadata

        # Test the case where the metadata and name is overridden in the function
        # call.
        new_collection = TargetTextCollection(self._target_text_examples()[:2], 
                                              name=name, metadata=metadata)
        json_multi_collection = new_collection.to_json()
        new_name = 'new name'
        new_metadata = {'model': 'TDLSTM'}
        from_json_collection = TargetTextCollection.from_json(json_multi_collection, 
                                                              name=new_name, metadata=new_metadata)
        assert new_collection == from_json_collection
        assert from_json_collection.name == new_name
        assert from_json_collection.metadata['name'] == new_name
        assert from_json_collection.metadata['model'] == 'TDLSTM'

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

        # Ensure that it can load the multiple target text instances with metadata
        # with and without a name variable
        for contains_name, metadata_fp in [(True, 'one_target_one_empty_metadata.json'), 
                                           (False, 'one_target_one_empty_metadata_no_name.json')]:
            two_target_metadata_json_fp = Path(self._json_data_dir(), metadata_fp)
            two_target_metadata_collection = TargetTextCollection.load_json(two_target_metadata_json_fp)
            assert len(two_target_collection) == 2
            correct_name = ''
            correct_metadata = {"target sentiment predictions": [{"model": "InterAE"}]}
            test_metadata = two_target_metadata_collection.metadata
            if contains_name:
                correct_name = "multi targets"
                correct_metadata['name'] = 'multi targets'
                assert correct_metadata['name'] == test_metadata['name']
            assert len(correct_metadata) == len(test_metadata)
            assert len(correct_metadata['target sentiment predictions']) == len(test_metadata['target sentiment predictions'])
            assert correct_metadata['target sentiment predictions'] == test_metadata['target sentiment predictions']
            assert two_target_metadata_collection.name == correct_name
            assert two_target_metadata_collection.metadata == correct_metadata

        # Ensure that when loading the given data is overriden
        two_target_metadata_json_fp = Path(self._json_data_dir(), 'one_target_one_empty_metadata.json')
        name = 'different'
        metadata = {'model': 'TDLSTM'}
        two_target_metadata_collection = TargetTextCollection.load_json(two_target_metadata_json_fp, name=name, metadata=metadata)
        assert two_target_metadata_collection.name == name
        assert two_target_metadata_collection.metadata == metadata
        assert 2 == len(two_target_metadata_collection)

    @pytest.mark.parametrize("name", (' ', 'test_name'))
    @pytest.mark.parametrize("metadata", (None, {'model': 'InterAE'}))
    def test_to_json_file(self, name: str, metadata: Optional[Dict[str, Any]]):
        if metadata is None:
            metadata = {}
        metadata['name'] = name
        with tempfile.NamedTemporaryFile(mode='w+') as temp_fp:
            temp_path = Path(temp_fp.name)
            test_collection = TargetTextCollection(name=name, metadata=metadata)
            test_collection.to_json_file(temp_path)
            loaded_collection = TargetTextCollection.load_json(temp_path)
            assert len(loaded_collection) == 0
            assert name == loaded_collection.name
            assert metadata == loaded_collection.metadata

            # Ensure that it can load more than one Target Text examples
            test_collection = TargetTextCollection(self._target_text_examples(), 
                                                   name=name, metadata=metadata)
            test_collection.to_json_file(temp_path)
            loaded_collection = TargetTextCollection.load_json(temp_path)
            assert len(loaded_collection) == 3
            assert name == loaded_collection.name
            assert metadata == loaded_collection.metadata

            # Ensure that if it saves to the same file that it overwrites that 
            # file
            test_collection = TargetTextCollection(self._target_text_examples(),
                                                   name=name, metadata=metadata)
            test_collection.to_json_file(temp_path)
            loaded_collection = TargetTextCollection.load_json(temp_path)
            assert len(loaded_collection) == 3
            assert name == loaded_collection.name
            assert metadata == loaded_collection.metadata

            # Ensure that it can just load one examples
            test_collection = TargetTextCollection([self._target_text_example()],
                                                   name=name, metadata=metadata)
            test_collection.to_json_file(temp_path)
            loaded_collection = TargetTextCollection.load_json(temp_path)
            assert len(loaded_collection) == 1
            assert name == loaded_collection.name
            assert metadata == loaded_collection.metadata
    
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
        test_collection.pos_text(spacy_tagger())
        pos_answer = ['DET', 'NOUN', 'NOUN', 'VERB', 'ADJ', 'CCONJ', 'NOUN', 
                      'VERB', 'ADJ']
        assert test_collection['2']['pos_tags'] == pos_answer

        # Test the normal case with multiple TargetText Instance in the 
        # collection
        test_collection = TargetTextCollection(self._target_text_examples())
        test_collection.pos_text(spacy_tagger())
        assert test_collection['2']['pos_tags'] == pos_answer

        # Ensure that at least one error is raised but all of these tests are 
        # covered in the TargetText tests.
        with pytest.raises(TypeError):
            test_collection.pos_text(str.strip)

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

    @pytest.mark.parametrize("return_errors", (False, True))
    def test_sequence_labels(self, return_errors: bool):
        # Test the single case
        test_collection = TargetTextCollection([self._target_text_example()])
        test_collection.tokenize(spacy_tokenizer())
        returned_errors = test_collection.sequence_labels(return_errors)
        assert not returned_errors
        correct_sequence = ['O', 'B', 'I', 'O', 'O', 'O', 'B', 'O', 'O']
        assert test_collection['2']['sequence_labels'] == correct_sequence

        # Test the multiple case
        test_collection = TargetTextCollection(self._target_text_examples())
        test_collection.tokenize(spacy_tokenizer())
        returned_errors = test_collection.sequence_labels(return_errors)
        assert not returned_errors
        correct_sequence = ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O']
        assert test_collection['another_id']['sequence_labels'] == correct_sequence

        # Test the case where you can have multiple sequence labels
        test_collection = TargetTextCollection(self._target_text_examples())
        error_text = 'Mental health service budgets cut by 8% as referrals to '\
                     'community mental health teams rise nearly 20%: '\
                     'http://t.co/ezxPGrNfeG #bbcdp'
        error_case = TargetText(**{'text': error_text, 
                                   'text_id': '78895626198466562', 
                                   'targets': ['health service', 'mental health', 
                                               'budgets', 'Mental health'], 
                                   'spans': [Span(start=7, end=21), 
                                             Span(start=66, end=79), 
                                             Span(start=22, end=29), 
                                             Span(start=0, end=13)], 
                                   'target_sentiments': ['neutral', 'neutral', 
                                                         'neutral', 'neutral'], 
                                   'categories': None, 'category_sentiments': None})
        test_collection.add(error_case)
        test_collection.tokenize(spacy_tokenizer())
        if not return_errors:
            with pytest.raises(ValueError):
                test_collection.sequence_labels(return_errors)
        else:
            returned_errors = test_collection.sequence_labels(return_errors)
            assert len(returned_errors) == 1
            assert returned_errors[0]['text_id'] == '78895626198466562'


    def test_exact_match_score(self):
        # Simple case where it should get perfect score
        test_collection = TargetTextCollection([self._target_text_example()])
        test_collection.tokenize(spacy_tokenizer())
        test_collection.sequence_labels()
        measures = test_collection.exact_match_score('sequence_labels')
        for index, measure in enumerate(measures):
            if index == 3:
                assert measure['FP'] == []
                assert measure['FN'] == []
                assert measure['TP'] == [('2', Span(4, 15)), ('2', Span(30, 35))]
            else:
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
        recall, precision, f1, error_analysis = test_collection.exact_match_score('sequence_labels')
        assert precision == 1.0
        assert recall == 2.0/3.0
        assert f1 == 0.8
        assert error_analysis['FP'] == []
        assert error_analysis['FN'] == [('0', Span(4, 15))]
        assert error_analysis['TP'] == [('0', Span(30, 35)), ('1', Span(4, 16))]

        # Something that has perfect recall but not precision as it over 
        # predicts 
        sequence_labels_0 = ['O', 'B', 'I', 'B', 'O', 'O', 'B', 'O', 'O']
        test_collection['0']['sequence_labels'] = sequence_labels_0
        sequence_labels_1 = ['O', 'B', 'I', 'O', 'O']
        test_collection['1']['sequence_labels'] = sequence_labels_1
        recall, precision, f1, error_analysis = test_collection.exact_match_score(
            'sequence_labels')
        assert precision == 3/4
        assert recall == 1.0
        assert round(f1, 3) == 0.857
        assert error_analysis['FP'] == [('0', Span(16, 19))]
        assert error_analysis['FN'] == []
        assert error_analysis['TP'] == [('0', Span(4, 15)), ('0', Span(30, 35)), 
                                        ('1', Span(4, 16))]

        # Does not predict anything for a whole sentence therefore will have 
        # perfect precision but bad recall (mainly testing the if not 
        # getting anything for a sentence matters)
        sequence_labels_0 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        test_collection['0']['sequence_labels'] = sequence_labels_0
        sequence_labels_1 = ['O', 'B', 'I', 'O', 'O']
        test_collection['1']['sequence_labels'] = sequence_labels_1
        recall, precision, f1, error_analysis = test_collection.exact_match_score(
            'sequence_labels')
        assert precision == 1.0
        assert recall == 1/3
        assert f1 == 0.5
        assert error_analysis['FP'] == []
        fn_error =  sorted(error_analysis['FN'], key=lambda x: x[1].start)
        assert fn_error == [('0', Span(4, 15)), ('0', Span(30, 35))]
        assert error_analysis['TP'] == [('1', Span(4, 16))]

        # Handle the edge case of not getting anything
        sequence_labels_0 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        test_collection['0']['sequence_labels'] = sequence_labels_0
        sequence_labels_1 = ['O', 'O', 'O', 'O', 'O']
        test_collection['1']['sequence_labels'] = sequence_labels_1
        recall, precision, f1, error_analysis = test_collection.exact_match_score('sequence_labels')
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0
        assert error_analysis['FP'] == []
        fn_error =  sorted(error_analysis['FN'], key=lambda x: x[1].start)
        assert fn_error == [('0', Span(4, 15)), ('1', Span(4, 16)),
                            ('0', Span(30, 35))]
        assert error_analysis['TP'] == []

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
        recall, precision, f1, error_analysis = test_collection.exact_match_score('sequence_labels')
        assert recall == 3/4
        assert precision == 3/4
        assert f1 == 0.75
        assert error_analysis['FP'] == [('inf', Span(4, 16))]
        assert error_analysis['FN'] == [('inf', Span(4, 15))]
        tp_error =  sorted(error_analysis['TP'], key=lambda x: x[1].start)
        assert tp_error == [('0', Span(4, 15)), ('1', Span(4, 16)),
                            ('0', Span(30, 35))]

        # This time it can get a perfect score as the token alignment will be 
        # perfect
        test_collection.tokenize(spacy_tokenizer())
        sequence_labels_align = ['O', 'B', 'I', 'O', 'O', 'O']
        test_collection['inf']['sequence_labels'] = sequence_labels_align
        recall, precision, f1, error_analysis = test_collection.exact_match_score('sequence_labels')
        assert recall == 1.0
        assert precision == 1.0
        assert f1 == 1.0
        assert error_analysis['FP'] == []
        assert error_analysis['FN'] == []
        tp_error =  sorted(error_analysis['TP'], key=lambda x: x[1].end)
        assert tp_error == [('0', Span(4, 15)), ('inf', Span(4, 15)),('1', Span(4, 16)),
                            ('0', Span(30, 35))]

        # Handle the case where one of the samples has no spans
        test_example = TargetText(text="I've had a bad day", text_id='50')
        other_examples = self._target_text_measure_examples()
        other_examples.append(test_example)
        test_collection = TargetTextCollection(other_examples)
        test_collection.tokenize(str.split)
        test_collection.sequence_labels()
        measures = test_collection.exact_match_score('sequence_labels')
        for index, measure in enumerate(measures):
            if index == 3:
                assert measure['FP'] == []
                assert measure['FN'] == []
                tp_error =  sorted(measure['TP'], key=lambda x: x[1].end)
                assert tp_error == [('0', Span(4, 15)), ('1', Span(4, 16)),
                                    ('0', Span(30, 35))]
            else:
                assert measure == 1.0
        # Handle the case where on the samples has no spans but has predicted 
        # there is a span there
        test_collection['50']['sequence_labels'] = ['B', 'I', 'O', 'O', 'O']
        recall, precision, f1, error_analysis = test_collection.exact_match_score('sequence_labels')
        assert recall == 1.0
        assert precision == 3/4
        assert round(f1, 3) == 0.857
        assert error_analysis['FP'] == [('50', Span(start=0, end=8))]
        assert error_analysis['FN'] == []
        tp_error =  sorted(error_analysis['TP'], key=lambda x: x[1].end)
        assert tp_error == [('0', Span(4, 15)), ('1', Span(4, 16)),
                            ('0', Span(30, 35))]
        # See if it can handle a collection that only contains no spans
        test_example = TargetText(text="I've had a bad day", text_id='50')
        test_collection = TargetTextCollection([test_example])
        test_collection.tokenize(str.split)
        test_collection.sequence_labels()
        measures = test_collection.exact_match_score('sequence_labels')
        for index, measure in enumerate(measures):
            if index == 3:
                assert measure['FP'] == []
                assert measure['FN'] == []
                assert measure['TP'] == []
            else:
                assert measure == 0.0
        # Handle the case the collection contains one spans but a mistake
        test_collection['50']['sequence_labels'] = ['B', 'I', 'O', 'O', 'O']
        measures = test_collection.exact_match_score('sequence_labels')
        for index, measure in enumerate(measures):
            if index == 3:
                assert measure['FP'] == [('50', Span(0, 8))]
                assert measure['FN'] == []
                assert measure['TP'] == []
            else:
                assert measure == 0.0
        # Should raise a KeyError if one of the TargetText instances does 
        # not have a Span key
        del test_collection['50']._storage['spans']
        with pytest.raises(KeyError):
            test_collection.exact_match_score('sequence_labels')
        # should raise a KeyError if one of the TargetText instances does 
        # not have a predicted sequence key
        test_collection = TargetTextCollection([self._target_text_example()])
        test_collection.tokenize(spacy_tokenizer())
        test_collection.sequence_labels()
        with pytest.raises(KeyError):
            measures = test_collection.exact_match_score('nothing')
        
        # Should raise a ValueError if there are multiple same true spans
        a = TargetText(text='hello how are you I am good', text_id='1',
                       targets=['hello','hello'], spans=[Span(0,5), Span(0,5)]) 
        test_collection = TargetTextCollection([a])
        test_collection.tokenize(str.split)
        test_collection['1']['sequence_labels'] = ['B', 'O', 'O', 'O', 'O', 'O', 'O']
        with pytest.raises(ValueError):
            test_collection.exact_match_score('sequence_labels')


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

    @pytest.mark.parametrize("lower", (False, True))
    @pytest.mark.parametrize("incl_none_targets", (False, True))
    def test_target_count_andnumber_targets(self, lower: bool,
                                            incl_none_targets: bool):
        # Start with an empty collection
        test_collection = TargetTextCollection()
        nothing = test_collection.target_count(lower)
        assert len(nothing) == 0
        assert not nothing

        # Collection that contains TargetText instances but with no targets
        test_collection.add(TargetText(text='some text', text_id='1'))
        assert len(test_collection) == 1
        nothing = test_collection.target_count(lower)
        assert len(nothing) == 0
        assert not nothing

        assert test_collection.number_targets(incl_none_targets) == 0

        # Collection now contains at least one target
        test_collection.add(TargetText(text='another item today', text_id='2',
                                       spans=[Span(0, 12)], 
                                       targets=['another item']))
        assert len(test_collection) == 2
        one = test_collection.target_count(lower)
        assert len(one) == 1
        assert one == {'another item': 1}

        assert test_collection.number_targets(incl_none_targets) == 1

        # Collection now contains 3 targets but 2 are the same
        test_collection.add(TargetText(text='another item today', text_id='3',
                                       spans=[Span(0, 12)], 
                                       targets=['another item']))
        test_collection.add(TargetText(text='item today', text_id='4',
                                       spans=[Span(0, 4)], 
                                       targets=['item']))
        assert len(test_collection) == 4
        two = test_collection.target_count(lower)
        assert len(two) == 2
        assert two == {'another item': 2, 'item': 1}

        assert test_collection.number_targets(incl_none_targets) == 3

        # Difference between lower being False and True
        test_collection.add(TargetText(text='Item today', text_id='5',
                                       spans=[Span(0, 4)], 
                                       targets=['Item']))
        assert len(test_collection) == 5
        three = test_collection.target_count(lower)
        if lower:
            assert len(three) == 2
            assert three == {'another item': 2, 'item': 2}
        else:
            assert len(three) == 3
            assert three == {'another item': 2, 'item': 1, 'Item': 1}

        assert test_collection.number_targets(incl_none_targets) == 4

        # Can be the case where the target is None and we do not want to 
        # include these I don't thinkk
        test_collection.add(TargetText(text='Item today', text_id='6',
                                       spans=[Span(0, 0)], 
                                       targets=[None]))
        assert len(test_collection) == 6
        three_alt = test_collection.target_count(lower)
        if lower:
            assert len(three_alt) == 2
            assert three_alt == {'another item': 2, 'item': 2}
        else:
            assert len(three_alt) == 3
            assert three_alt == {'another item': 2, 'item': 1, 'Item': 1}

        if incl_none_targets:
            assert test_collection.number_targets(incl_none_targets) == 5
        else:
            assert test_collection.number_targets(incl_none_targets) == 4

    def test_category_count_and_numbers(self):
        # Start with an empty collection
        test_collection = TargetTextCollection()
        nothing = test_collection.number_categories()
        assert 0 == nothing
        assert {} == test_collection.category_count()

        # Collection that contains TargetText instances but with no categories
        test_collection.add(TargetText(text='some text', text_id='1'))
        assert len(test_collection) == 1
        nothing = test_collection.number_categories()
        assert 0 == nothing
        assert {} == test_collection.category_count()

        # Collection now contains at least one categories
        test_collection.add(TargetText(text='another item today', text_id='2', 
                                       categories=['ITEM']))
        assert len(test_collection) == 2
        assert 1 == test_collection.number_categories()
        correct_categories = {'ITEM': 1}
        for category, count in test_collection.category_count().items():
            assert correct_categories[category] == count
        assert len(correct_categories) == len(test_collection.category_count())

        # Collection now contains 5 categories but 3 are the same
        test_collection.add(TargetText(text='another item today', text_id='3', 
                                       categories=['ITEM', 'ITEM']))
        test_collection.add(TargetText(text='item today', text_id='4',
                                       categories=['SOMETHING', 'ANOTHER']))
        assert len(test_collection) == 4
        assert 5 == test_collection.number_categories()
        correct_categories = {'ITEM': 3, 'SOMETHING': 1, 'ANOTHER': 1}
        for category, count in test_collection.category_count().items():
            assert correct_categories[category] == count
        assert len(correct_categories) == len(test_collection.category_count())

        # Test the error case that the a category value is None
        test_collection.add(TargetText(text='item today', text_id='6',
                                       categories=[None]))
        with pytest.raises(ValueError):
            test_collection.number_categories()
        with pytest.raises(ValueError):
            test_collection.category_count()

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
    
    def test_combine(self):
        targets = [TargetText(text='some text', text_id='1')]
        # The case of one collection
        test_collection = TargetTextCollection(targets)

        combined_collection = TargetTextCollection.combine(test_collection)
        assert len(combined_collection) == len(test_collection)
        assert combined_collection == test_collection

        targets_1 = [TargetText(text='some text', text_id='2'),
                     TargetText(text='some text', text_id='3')]
        # The case of two collections
        test_collection_1 = TargetTextCollection(targets_1)

        combined_collection = TargetTextCollection.combine(test_collection, test_collection_1)
        assert len(combined_collection) == 3
        correct_combined = TargetTextCollection(targets + targets_1)
        assert combined_collection == correct_combined

        # Test the case of where two collections have a TargetText with the 
        # same key
        target_same_key = [TargetText(text='another', text_id='1')]
        test_collection_2 = TargetTextCollection(target_same_key)
        combined_collection = TargetTextCollection.combine(test_collection, test_collection_1, 
                                                           test_collection_2)
        assert len(combined_collection) == 3
        assert combined_collection['1']['text'] == 'another'
        correct_combined = TargetTextCollection(targets_1 + target_same_key)
        assert combined_collection == correct_combined

        # Test the case of combining two empty TargetTextCollections
        empty_collection = TargetTextCollection()
        empty_collection_1 = TargetTextCollection()
        empty_combined = TargetTextCollection.combine(empty_collection, empty_collection_1)
        assert len(empty_combined) == 0
        assert empty_combined == empty_collection
        assert empty_combined == empty_collection_1

    @pytest.mark.parametrize("lower", (False, True))
    @pytest.mark.parametrize("unique_sentiment", (False, True))
    def test_target_sentiments(self, lower: bool, unique_sentiment: bool):
        # Test the empty case
        test_collection = TargetTextCollection()
        nothing = test_collection.target_sentiments(lower, unique_sentiment)
        assert len(nothing) == 0
        assert not nothing

        # Collection that contains TargetText instances but with no targets
        test_collection.add(TargetText(text='some text', text_id='1'))
        assert len(test_collection) == 1
        nothing = test_collection.target_sentiments(lower, unique_sentiment)
        assert len(nothing) == 0
        assert not nothing

        # Test the case where targets exist but no sentiments
        test_collection.add(TargetText(text='another item today', text_id='2',
                                       spans=[Span(0, 12)], 
                                       targets=['another item']))
        assert len(test_collection) == 2
        nothing = test_collection.target_sentiments(lower, unique_sentiment)
        assert len(nothing) == 0
        assert not nothing

        # Test the single target and single sentiment case
        test_collection['2']['target_sentiments'] = ['positive']
        assert len(test_collection) == 2
        one = test_collection.target_sentiments(lower, unique_sentiment)
        if unique_sentiment:
            assert {'another item': {'positive'}} == one
        else:
            assert {'another item': ['positive']} == one

        # Test the case with more than one target but only one sentiment
        test_collection.add(TargetText(text='item today', text_id='4',
                                       spans=[Span(0, 4)],
                                       target_sentiments=['negative'], 
                                       targets=['item']))
        assert len(test_collection) == 3
        two = test_collection.target_sentiments(lower, unique_sentiment)
        if unique_sentiment:
            correct = {'another item': {'positive'}, 'item': {'negative'}}
        else:
            correct = {'another item': ['positive'], 'item': ['negative']}
        for key, value in correct.items():
            assert value == two[key]
        assert len(correct) == len(two)
        # Test the case where there are three targets but two sentiments
        test_collection.add(TargetText(text='Item today', text_id='5',
                                       spans=[Span(0, 4)], 
                                       targets=['Item']))
        assert len(test_collection) == 4
        two = test_collection.target_sentiments(lower, unique_sentiment)
        for key, value in correct.items():
            assert value == two[key]
        assert len(correct) == len(two)
        # Test the case where in the case sensitive there are three targets
        # in the non case sensitive there are two
        test_collection['5']['target_sentiments'] = ['negative']
        two = test_collection.target_sentiments(lower, unique_sentiment)
        if lower:
            correct = {'another item': ['positive'], 
                       'item': ['negative', 'negative']}
            if unique_sentiment:
                correct['item'] = {'negative'}
                correct['another item'] = {'positive'}
        else:
            correct = {'another item': ['positive'], 
                       'item': ['negative'],
                       'Item': ['negative']}
            if unique_sentiment:
                correct = {'another item': {'positive'}, 
                           'item': {'negative'},
                           'Item': {'negative'}}
        for key, value in correct.items():
            assert value == two[key]
        assert len(correct) == len(two)
        # Test the case where you can have multiple sentiment for one target
        test_collection.add(TargetText(text='great Another item', text_id='6',
                                       spans=[Span(6, 18)], 
                                       target_sentiments=['negative'],
                                       targets=['Another item']))
        target_sentiment = test_collection.target_sentiments(lower, unique_sentiment)
        if lower:
            correct = {'another item': ['negative', 'positive'], 
                       'item': ['negative', 'negative']}
            if unique_sentiment:
                correct['item'] = {'negative'}
                correct['another item'] = {'positive', 'negative'}
        else:
            correct = {'another item': ['positive'],
                       'Another item': ['negative'], 
                       'item': ['negative'],
                       'Item': ['negative']}
            if unique_sentiment:
                correct['item'] = {'negative'}
                correct['Item'] = {'negative'}
                correct['Another item'] = {'negative'}
                correct['another item'] = {'positive'}
        for key, value in correct.items():
            assert sorted(value) == sorted(target_sentiment[key])
        assert len(correct) == len(target_sentiment)
        # Test the case where one of the targets in the TargetCollection contains
        # multiple targets and target sentiments as well as None value
        test_collection.add(TargetText(text='great day but food was bad and the table', text_id='7',
                                       spans=[Span(6, 9), Span(0, 0), Span(14, 18), Span(35, 41)], 
                                       target_sentiments=['negative', 'positive', 'negative', 'neutral'],
                                       targets=['day', None, 'food', 'table']))
        target_sentiment = test_collection.target_sentiments(lower, unique_sentiment)
        if lower:
            correct = {'another item': ['negative', 'positive'], 
                       'item': ['negative', 'negative'],
                       'day': ['negative'], 'food': ['negative'], 
                       'table': ['neutral']}
            if unique_sentiment:
                correct['item'] = {'negative'}
                correct['another item'] = {'positive', 'negative'}
                correct['table'] = {'neutral'}
                correct['day'] = {'negative'}
                correct['food'] = {'negative'}
        else:
            correct = {'another item': ['positive'],
                       'Another item': ['negative'], 
                       'item': ['negative'],'day': ['negative'], 
                       'food': ['negative'], 'table': ['neutral'],
                       'Item': ['negative']}
            if unique_sentiment:
                correct['item'] = {'negative'}
                correct['another item'] = {'positive'}
                correct['Another item'] = {'negative'}
                correct['Item'] = {'negative'}
                correct['table'] = {'neutral'}
                correct['day'] = {'negative'}
                correct['food'] = {'negative'}
        for key, value in correct.items():
            assert sorted(value) == sorted(target_sentiment[key])
        assert len(correct) == len(target_sentiment)

    @pytest.mark.parametrize("sentiment_key", ('target_sentiments', 'test_name'))
    def test_unique_distinct_sentiments(self, sentiment_key: str):
        text = 'The laptop case was great and cover was rubbish'
        text_id = '0'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        sentiments = {sentiment_key: [1, 2]}
        target_text_0 = TargetText(text=text, text_id=text_id, spans=spans, 
                                   targets=targets, **sentiments)
        text = 'The laptop price was awful'
        text_id = '1'
        spans = [Span(4, 16)]
        targets = ['laptop price']
        sentiments = {sentiment_key: [1]}
        target_text_1 = TargetText(text=text, text_id=text_id, spans=spans,
                                   targets=targets, **sentiments)
        text = 'The laptop case was great and cover was rubbish'
        text_id = '2'
        spans = [Span(4, 15), Span(30, 35)]
        targets = ['laptop case', 'cover']
        sentiments = {sentiment_key: [3, 3]}
        target_text_2 = TargetText(text=text, text_id=text_id, spans=spans, 
                                   targets=targets, **sentiments)
        text = 'The laptop case was great and cover was rubbish'
        text_id = '3'
        spans = []
        targets = []
        sentiments = {sentiment_key: []}
        target_text_3 = TargetText(text=text, text_id=text_id, spans=spans, 
                                   targets=targets, **sentiments)
        # 0 DS
        coll_0 = TargetTextCollection([target_text_3])
        assert set() == coll_0.unique_distinct_sentiments(sentiment_key)
        # 1 DS
        coll_1 = TargetTextCollection([target_text_2])
        assert set([1]) == coll_1.unique_distinct_sentiments(sentiment_key)
        coll_1 = TargetTextCollection([target_text_1])
        assert set([1]) == coll_1.unique_distinct_sentiments(sentiment_key)
        # 2 DS
        coll_2 = TargetTextCollection([target_text_0])
        assert set([2]) == coll_2.unique_distinct_sentiments(sentiment_key)
        coll_2 = TargetTextCollection([target_text_0, target_text_3])
        assert set([2]) == coll_2.unique_distinct_sentiments(sentiment_key)
        # 2 and 1 DS
        coll_2_1 = TargetTextCollection([target_text_0, target_text_1])
        assert set([1,2]) == coll_2_1.unique_distinct_sentiments(sentiment_key)
        coll_2_1 = TargetTextCollection([target_text_0, target_text_2])
        assert set([1,2]) == coll_2_1.unique_distinct_sentiments(sentiment_key)
        coll_2_1 = TargetTextCollection([target_text_0, target_text_1])
        assert set([1,2]) == coll_2_1.unique_distinct_sentiments(sentiment_key)
        # Raises TypeError if the sentiment value is not of type list
        text = 'The laptop case was great and cover was rubbish'
        text_id = '4'
        spans = []
        targets = []
        sentiments = {sentiment_key: None}
        target_text_4 = TargetText(text=text, text_id=text_id, spans=spans, 
                                   targets=targets, **sentiments)
        with pytest.raises(TypeError):
            coll_err = TargetTextCollection([target_text_4])
            coll_err.unique_distinct_sentiments(sentiment_key)