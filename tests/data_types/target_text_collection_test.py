from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile

import pytest

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.data_types_util import Span, AnonymisedError, OverwriteError
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

        # Test the anonymised case
        assert not example_collection.anonymised
        example_collection.anonymised = True
        example_collection['1'] = TargetText(text='something', text_id='1')
        assert 'text' not in example_collection['2']
        assert 'text' not in example_collection['1']

    def test_add(self):
        new_collection = TargetTextCollection()

        assert len(new_collection) == 0
        new_collection.add(self._target_text_example())
        assert len(new_collection) == 1

        assert '2' in new_collection
        assert not new_collection.anonymised

        # Test the anonymised case
        new_collection.anonymised = True
        assert new_collection.anonymised
        new_collection.add(TargetText(text='something', text_id='1'))
        assert '1' in new_collection
        assert 'text' not in new_collection['2']
        assert 'text' not in new_collection['1']

    def test_construction(self):
        new_collection = TargetTextCollection(name='new_name')
        assert new_collection.name == 'new_name'
        assert new_collection.metadata == {'name': 'new_name'}

        new_collection = TargetTextCollection()
        assert new_collection.name == ''
        assert new_collection.metadata == {'name': ''}

        example_metadata = {'model': 'InterAE', 'name': ''}
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
        assert new_collection.metadata == {'name': ''}
        assert new_collection.name == ''
        assert new_collection.anonymised == False
        assert 'text' in new_collection['0']

        # Test anonymised version
        new_collection = TargetTextCollection(target_texts=self._target_text_examples(), 
                                              anonymised=True)
        assert new_collection.anonymised == True
        assert new_collection.metadata == {'anonymised': True, 'name': ''}
        assert 'text' not in new_collection['0']

    def test_name(self):
        new_collection = TargetTextCollection(target_texts=self._target_text_examples())
        assert new_collection.name == ''
        assert new_collection.metadata == {'name': ''}
        new_collection.name = 'something new'
        assert new_collection.name == 'something new'
        assert new_collection.metadata == {'name': 'something new'}

    def test_anonymised(self):
        new_collection = TargetTextCollection(target_texts=self._target_text_examples())
        assert not new_collection.anonymised
        for target_text in new_collection.values():
            assert 'text' in target_text
        assert new_collection.metadata == {'name': ''}
        new_collection.anonymised = True
        assert new_collection.anonymised
        correct_metadata = {'name': '', 'anonymised': True}
        assert len(new_collection.metadata) == len(correct_metadata)
        for key, value in correct_metadata.items():
            assert value == new_collection.metadata[key]
        for target_text in new_collection.values():
            assert 'text' not in target_text
        with pytest.raises(AnonymisedError):
            new_collection.anonymised = False
        assert new_collection.anonymised
        assert len(new_collection.metadata) == len(correct_metadata)
        for key, value in correct_metadata.items():
            assert value == new_collection.metadata[key]


    def test_to_json(self):
        # no target text instances in the collection (empty collection)
        new_collection = TargetTextCollection()
        assert new_collection.to_json() == '{"metadata": {"name": ""}}'

        # One target text in the collection
        new_collection = TargetTextCollection([self._target_text_example()])
        true_json_version = ('{"text": "The laptop case was great and cover '
                             'was rubbish", "text_id": "2", "targets": ["laptop '
                             'case", "cover"], "spans": [[4, 15], [30, 35]], '
                             '"target_sentiments": [0, 1], "categories": '
                             '["LAPTOP#CASE", "LAPTOP"], "category_sentiments": null}\n'
                             '{"metadata": {"name": ""}}')
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
                             '["LAPTOP"], "category_sentiments": null}\n'
                             '{"metadata": {"name": ""}}')
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
                             '{"metadata": {"model": "InterAE", "name": ""}}')
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

        # Test that anonymised is added
        new_collection = TargetTextCollection(self._target_text_examples()[:2], 
                                              anonymised=True)
        true_json_version = ('{"text_id": "0", "targets": '
                             '["laptop case"], "spans": [[4, 15]], '
                             '"target_sentiments": [0], "categories": '
                             '["LAPTOP#CASE"], "category_sentiments": null}'
                             '\n{"text_id": '
                             '"another_id", "targets": ["cover"], "spans": '
                             '[[30, 35]], "target_sentiments": [1], "categories": '
                             '["LAPTOP"], "category_sentiments": null}\n'
                             '{"metadata": {"anonymised": true, "name": ""}}')
        assert new_collection.to_json() == true_json_version

    @pytest.mark.parametrize("anonymised", (True, False))
    @pytest.mark.parametrize("name", (' ', 'test_name'))
    @pytest.mark.parametrize("metadata", (None, 'InterAE'))
    def test_from_json(self, name: str, metadata: Optional[Dict[str, Any]],
                       anonymised: bool):
        if metadata == 'InterAE':
            metadata = {'model': metadata}
        if metadata is None:
            metadata = {}
            metadata['name'] = name
            if anonymised:
                metadata['anonymised'] = anonymised
        # no text given
        test_collection = TargetTextCollection.from_json('', name=name, 
                                                         metadata=metadata,
                                                         anonymised=anonymised)
        assert TargetTextCollection() == test_collection
        assert test_collection.name == name
        assert len(test_collection.metadata) == len(metadata)
        for key, value in metadata.items():
            assert value == test_collection.metadata[key]
        assert test_collection.anonymised == anonymised

        # One target text instance in the text
        new_collection = TargetTextCollection([self._target_text_example()], 
                                              name=name, metadata=metadata,
                                              anonymised=anonymised)
        json_one_collection = new_collection.to_json()
        assert new_collection == TargetTextCollection.from_json(json_one_collection)
        assert new_collection.name == name
        assert new_collection.metadata == metadata
        assert len(new_collection.metadata) == len(metadata)
        for key, value in metadata.items():
            assert value == new_collection.metadata[key]
        assert new_collection.anonymised == anonymised

        # Multiple target text instances in the text
        new_collection = TargetTextCollection(self._target_text_examples()[:2], 
                                              name=name, metadata=metadata,
                                              anonymised=anonymised)
        json_multi_collection = new_collection.to_json()
        assert new_collection == TargetTextCollection.from_json(json_multi_collection)
        assert new_collection.name == name
        assert len(new_collection.metadata) == len(metadata)
        for key, value in metadata.items():
            assert value == new_collection.metadata[key]
        assert new_collection.anonymised == anonymised

        # Test the case where the metadata, name, and anonymised is overridden
        # in the function call.
        new_collection = TargetTextCollection(self._target_text_examples()[:2], 
                                              name=name, metadata=metadata,
                                              anonymised=anonymised)
        json_multi_collection = new_collection.to_json()
        new_name = 'new name'
        new_metadata = {'model': 'TDLSTM'}
        new_anonymised = not anonymised
        if new_anonymised == False and anonymised:
            with pytest.raises(AnonymisedError):
                TargetTextCollection.from_json(json_multi_collection, name=new_name, 
                                            metadata=new_metadata, anonymised=new_anonymised)
        else:
            from_json_collection = TargetTextCollection.from_json(json_multi_collection, 
                                                                name=new_name, metadata=new_metadata,
                                                                anonymised=new_anonymised)
            assert new_collection == from_json_collection
            assert from_json_collection.name == new_name
            assert from_json_collection.metadata['name'] == new_name
            assert from_json_collection.metadata['model'] == 'TDLSTM'
            assert from_json_collection.metadata['anonymised'] == True
            assert 'text' not in from_json_collection['0']

        
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
        # with and without a name variable.
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
            correct_metadata['name'] = correct_name
            assert correct_metadata['name'] == test_metadata['name']
            assert len(correct_metadata) == len(test_metadata)
            assert len(correct_metadata['target sentiment predictions']) == len(test_metadata['target sentiment predictions'])
            assert correct_metadata['target sentiment predictions'] == test_metadata['target sentiment predictions']
            assert two_target_metadata_collection.name == correct_name
            assert len(two_target_metadata_collection.metadata) == len(correct_metadata)
            for key, value in correct_metadata.items():
                assert value == two_target_metadata_collection.metadata[key]
            assert not two_target_metadata_collection.anonymised

        # Ensure that when loading the given data is overriden
        two_target_metadata_json_fp = Path(self._json_data_dir(), 'one_target_one_empty_metadata.json')
        new_name = 'different'
        new_metadata = {'model': 'TDLSTM'}

        two_target_metadata_collection = TargetTextCollection.load_json(two_target_metadata_json_fp, 
                                                                        name=new_name, metadata=new_metadata,
                                                                        anonymised=True)
        new_metadata['name'] = new_name
        new_metadata['anonymised'] = True
        assert two_target_metadata_collection.name == new_name
        assert len(two_target_metadata_collection.metadata) == len(new_metadata)
        for key, value in new_metadata.items():
            assert value == two_target_metadata_collection.metadata[key]
        assert 2 == len(two_target_metadata_collection)
        assert two_target_metadata_collection.anonymised
        assert 'text' not in two_target_metadata_collection['1']

        # Test that it can load a dataset that has been anonymised
        anonymised_fp = Path(self._json_data_dir(), 'anonymised.json')
        loaded_dataset = TargetTextCollection.load_json(anonymised_fp)
        assert 2 == len(loaded_dataset)
        for target_object in loaded_dataset.values():
            assert 'text' not in target_object
        assert loaded_dataset.anonymised
        assert 'loaded' == loaded_dataset.name
        assert True == loaded_dataset.metadata['anonymised']

    @pytest.mark.parametrize("anonymised", (True, False))
    @pytest.mark.parametrize("name", (' ', 'test_name'))
    @pytest.mark.parametrize("metadata", (None, 'InterAE'))
    @pytest.mark.parametrize("include_metadata", (True, False))
    def test_to_json_file(self, name: str, metadata: Optional[Dict[str, Any]],
                          anonymised: bool, include_metadata: bool):
        if metadata == 'InterAE':
            metadata = {'model': metadata}
        if metadata is None:
            metadata = {}
        metadata['name'] = name
        if anonymised:
            metadata['anonymised'] = anonymised
        with tempfile.NamedTemporaryFile(mode='w+') as temp_fp:
            temp_path = Path(temp_fp.name)
            test_collection = TargetTextCollection(name=name, metadata=metadata, 
                                                   anonymised=anonymised)
            test_collection.to_json_file(temp_path, include_metadata=include_metadata)
            loaded_collection = TargetTextCollection.load_json(temp_path)
            assert len(loaded_collection) == 0
            if include_metadata:
                assert name == loaded_collection.name
                assert len(metadata) == len(loaded_collection.metadata)
                for key, value in metadata.items():
                    assert value == loaded_collection.metadata[key]
                assert loaded_collection.anonymised == anonymised

            # Ensure that it can load more than one Target Text examples
            test_collection = TargetTextCollection(self._target_text_examples(), 
                                                   name=name, metadata=metadata,
                                                   anonymised=anonymised)
            test_collection.to_json_file(temp_path, include_metadata=include_metadata)
            if include_metadata:
                loaded_collection = TargetTextCollection.load_json(temp_path)
                assert len(loaded_collection) == 3
                assert name == loaded_collection.name
                assert len(metadata) == len(loaded_collection.metadata)
                for key, value in metadata.items():
                    assert value == loaded_collection.metadata[key]
                assert loaded_collection.anonymised == anonymised
            # Cannot load anonymised data without knowing it is anonymised 
            # through the metadata
            elif not anonymised:
                loaded_collection = TargetTextCollection.load_json(temp_path)
                assert len(loaded_collection) == 3

            # Ensure that if it saves to the same file that it overwrites that 
            # file
            test_collection = TargetTextCollection(self._target_text_examples(),
                                                   name=name, metadata=metadata,
                                                   anonymised=anonymised)
            test_collection.to_json_file(temp_path, include_metadata=include_metadata)
            if include_metadata:
                loaded_collection = TargetTextCollection.load_json(temp_path)
                assert len(loaded_collection) == 3
                assert name == loaded_collection.name
                assert len(metadata) == len(loaded_collection.metadata)
                for key, value in metadata.items():
                    assert value == loaded_collection.metadata[key]
                assert loaded_collection.anonymised == anonymised
            elif not anonymised:
                loaded_collection = TargetTextCollection.load_json(temp_path)
                assert len(loaded_collection) == 3

            # Ensure that it can just load one examples
            test_collection = TargetTextCollection([self._target_text_example()],
                                                   name=name, metadata=metadata,
                                                   anonymised=anonymised)
            test_collection.to_json_file(temp_path, include_metadata=include_metadata)
            if include_metadata:
                loaded_collection = TargetTextCollection.load_json(temp_path)
                assert len(loaded_collection) == 1
                assert name == loaded_collection.name
                assert len(metadata) == len(loaded_collection.metadata)
                for key, value in metadata.items():
                    assert value == loaded_collection.metadata[key]
                assert loaded_collection.anonymised == anonymised
            elif not anonymised:
                loaded_collection = TargetTextCollection.load_json(temp_path)
                assert len(loaded_collection) == 1
    
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

    @pytest.mark.parametrize("label_key", (None, 'target_sentiments'))
    @pytest.mark.parametrize("return_errors", (False, True))
    def test_sequence_labels(self, return_errors: bool, label_key: str):
        # Test the single case
        test_collection = TargetTextCollection([self._target_text_example()])
        test_collection.tokenize(spacy_tokenizer())
        returned_errors = test_collection.sequence_labels(return_errors, 
                                                          label_key=label_key)
        assert not returned_errors
        correct_sequence = ['O', 'B', 'I', 'O', 'O', 'O', 'B', 'O', 'O']
        if label_key is not None:
            correct_sequence = ['O', 'B-0', 'I-0', 'O', 'O', 'O', 'B-1', 'O', 'O']
        assert test_collection['2']['sequence_labels'] == correct_sequence

        # Test the multiple case
        test_collection = TargetTextCollection(self._target_text_examples())
        test_collection.tokenize(spacy_tokenizer())
        returned_errors = test_collection.sequence_labels(return_errors, 
                                                          label_key=label_key)
        assert not returned_errors
        correct_sequence = ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O']
        if label_key is not None:
            correct_sequence = ['O', 'O', 'O', 'O', 'O', 'O', 'B-1', 'O', 'O']
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
                test_collection.sequence_labels(return_errors, 
                                                label_key=label_key)
        else:
            returned_errors = test_collection.sequence_labels(return_errors, 
                                                              label_key=label_key)
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
    def test_target_count_and_number_targets(self, lower: bool,
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
        # Should raise a KeyError if the `target_key` does not exist
        with pytest.raises(KeyError):
            test_collection.target_count(lower, target_key='does not exist')

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
        
    def test_de_anonymise(self):
        import copy
        target_examples = self._target_text_examples()
        copy_target_examples = iter(copy.deepcopy(target_examples))
        collection = TargetTextCollection(target_examples, anonymised=True)
        assert collection.anonymised
        for target in collection.values():
            assert 'text' not in target

        collection.de_anonymise(copy_target_examples)
        assert not collection.anonymised
        correct_text = 'The laptop case was great and cover was rubbish'
        for target in collection.values():
            assert 'text' in target
            assert correct_text == target['text']
            assert not target.anonymised
        # Test the case where the length of the text dictionaries given are not 
        # the same length as the TargetTextCollection
        target_examples = self._target_text_examples()
        collection = TargetTextCollection(target_examples, anonymised=True)
        wrong_dict_length = [{'text_id': '0', 'text': correct_text},
                             {'text_id': '0', 'text': correct_text},
                             {'text_id': '0', 'text': correct_text}]
        with pytest.raises(ValueError):
            collection.de_anonymise(wrong_dict_length)
        wrong_dict_length = [{'text_id': '0', 'text': correct_text},
                             {'text_id': '2', 'text': correct_text}]
        with pytest.raises(ValueError):
            collection.de_anonymise(wrong_dict_length)
        assert collection.anonymised
        for target in collection.values():
            assert 'text' not in target
            assert target.anonymised
        # Test the case where the length of the text dictionaries is correct 
        # but the dictionary keys are not correct.
        wrong_keys = [{'text_id': '0', 'text': correct_text}, 
                      {'text_id': '1', 'text': correct_text}, 
                      {'text_id': '2', 'text': correct_text}]
        with pytest.raises(KeyError):
            collection.de_anonymise(wrong_keys)
        for target in collection.values():
            assert 'text' not in target
            assert target.anonymised
        assert collection.anonymised
        # Test the case where the text is in-correct
        wrong_text = 'The laptop case was great and cove was rubbish'
        wrong_text_dicts = [{'text_id': '0', 'text': wrong_text}, 
                            {'text_id': '2', 'text': wrong_text}, 
                            {'text_id': 'another_id', 'text': wrong_text}]
        with pytest.raises(AnonymisedError):
            collection.de_anonymise(wrong_text_dicts)
        for target in collection.values():
            assert 'text' not in target
            assert target.anonymised
        assert collection.anonymised

    def test_in_order(self):
        assert TargetTextCollection(self._target_text_examples()).in_order()
        # not in order case
        examples = self._target_text_examples()
        example = examples[-1]
        valid_spans = example._storage['spans']
        valied_targets = example._storage['targets']
        example._storage['spans'] = [valid_spans[1], valid_spans[0]]
        example._storage['targets'] = [valied_targets[1], valied_targets[0]]
        examples.pop()
        examples.append(example)
        assert not TargetTextCollection(examples).in_order()

        # Test the case where two targets overlap and they start in the same 
        # position this should return in_order
        edge_case = TargetText(text_id='1', text='had a good day', 
                               targets=['good', 'good day'], 
                               spans=[Span(6,10), Span(6,14)])
        edge_case.sanitize()
        assert TargetTextCollection([edge_case]).in_order()
    
    def test_re_order(self):
        examples = self._target_text_examples()
        example = examples[-1]
        valid_spans = example._storage['spans']
        valied_targets = example._storage['targets']
        example._storage['spans'] = [valid_spans[1], valid_spans[0]]
        example._storage['targets'] = [valied_targets[1], valied_targets[0]]
        examples.pop()
        examples.append(example)
        collection = TargetTextCollection(examples)
        assert not collection.in_order()
        collection.re_order()
        assert collection.in_order()
        # Test that it can be re-ordered twice without any affect
        collection.re_order()
        assert collection.in_order()
        # Test the rollback case
        examples = TargetTextCollection(self._target_text_examples())
        del examples['2']
        example = self._target_text_examples()[-1]
        valid_spans = example._storage['spans']
        valied_targets = example._storage['targets']
        example._storage['spans'] = [valid_spans[1], valid_spans[0]]
        example._storage['targets'] = [valied_targets[1], valied_targets[0]]
        examples.add(example)
        examples.add(TargetText(text_id='something', text=None, spans=[], tokens=['hello']))
        
        with pytest.raises(Exception):
            examples.re_order()
        assert Span(30, 35) == examples['2']['spans'][0]

    def test_add_unique_key(self):
        examples = self._target_text_examples()
        text_ids = ['0', 'another_id', '2']
        collection = TargetTextCollection(examples)
        collection.add_unique_key('spans', 'span_ids')
        for index, value in enumerate(collection.values()):
            correct_ids = []
            for key_index in range(len(value['spans'])):
                correct_ids.append(f'{text_ids[index]}::{key_index}')
            assert correct_ids == value['span_ids']

    def add_keys_and_values(self, collection: 'TargetTextCollection', 
                            key_to_values: List[List[str]],
                            values_to_add: List[List[List[Any]]]
                            ) -> 'TargetTextCollection':
        '''
        :param collection: The collection that is going to have the keys and 
                           associated values added to its collection.
        :param keys_to_values: A list of a list of keys where each key has to 
                               within the inner keys has to be repeated for 
                               number of targets
        :param values_to_add: A list of a list of a list of values where the 
                              outer most list represents the number of keys 
                              from `keys_to_values`, and the next inner list represents 
                              the number of targets within the collection, and then 
                              the inner most list represents the number of values 
                              to add for that Target.
        '''
        for target_index, target_text in enumerate(collection.values()):
            for keys_to_add in range(len(key_to_values)):
                added_key = key_to_values[keys_to_add][target_index]
                added_value = values_to_add[keys_to_add][target_index]
                target_text[added_key] = added_value
        return collection

    def test_key_difference(self):
        # Normal case where the given collection has all of the keys of the 
        # other collection but the other collection does have a few more keys 
        collection_1 = TargetTextCollection(self._target_text_examples())
        collection_2 = TargetTextCollection(self._target_text_examples())
        added_values = [[[1], [2], [3, 4]], [['great'], ['another'], ['better', 'that']]]
        added_keys = [['new_key'] * 3, ['another_key'] * 3]
        collection_2 = self.add_keys_and_values(collection_2, added_keys, added_values)
        correct_key_difference = ['new_key', 'another_key']
        differences = collection_1.key_difference(collection_2)
        assert len(correct_key_difference) == len(differences)
        for difference in correct_key_difference:
            assert difference in differences
        # The case where one collection contains all of the other collection
        assert not len(collection_2.key_difference(collection_1))

        # The case where both of them have key differences but each have 
        # different differences
        added_values = [[[1], [2], [3, 4]]]
        added_keys = [['different_key'] * 3]
        collection_1 = self.add_keys_and_values(collection_1, added_keys, added_values)
        collection_1_difference = ['new_key', 'another_key']
        differences = collection_1.key_difference(collection_2)
        assert len(collection_1_difference) == len(differences)
        for difference in collection_1_difference:
            assert difference in differences
        
        collection_2_difference = ['different_key']
        differences = collection_2.key_difference(collection_1)
        assert len(collection_2_difference) == len(differences)
        for difference in collection_2_difference:
            assert difference in differences
    
    @pytest.mark.parametrize("raise_on_overwrite", (True, False))
    @pytest.mark.parametrize("check_same_ids", (True, False))
    def test_combine_data_on_id(self, raise_on_overwrite: bool, 
                                check_same_ids: bool):
        # Normal case of copying from collection 2 to collection 1 where they 
        # both have the same ID's
        collection_1 = TargetTextCollection(self._target_text_examples())
        collection_1.add_unique_key('targets', 'target_id')
        collection_2 = TargetTextCollection(self._target_text_examples())
        collection_2.add_unique_key('targets', 'target_id')
        # values to add
        added_values = [[[1], [2], [3, 4]]]
        added_keys = [['different_key'] * 3]
        collection_2 = self.add_keys_and_values(collection_2, added_keys, added_values)
        assert ['different_key'] == collection_1.key_difference(collection_2)
        collection_1.combine_data_on_id(collection_2, 'target_id', ['different_key'],
                                        raise_on_overwrite=raise_on_overwrite, 
                                        check_same_ids=check_same_ids)
        for target_index, target_text in enumerate(collection_1.values()):
            for value_index, value in enumerate(target_text['different_key']):
                assert added_values[0][target_index][value_index] == value
        assert collection_1['2']['different_key'][1] == collection_2['2']['different_key'][1]
        assert collection_1['2']['target_id'][1] == collection_2['2']['target_id'][1]
        assert collection_1['2']['different_key'][0] == collection_2['2']['different_key'][0]
        assert collection_1['2']['target_id'][0] == collection_2['2']['target_id'][0]
        # The more difficult case where the 2nd collection contains ids that 
        # exist in both collections but id are different order
        temp_ids = collection_2['2']['target_id']
        temp_diff_values = collection_2['2']['different_key']
        collection_2['2']._storage['target_id'] = [temp_ids[1], temp_ids[0]]
        collection_2['2']._storage['different_key'] = [temp_diff_values[1], temp_diff_values[0]]
        collection_2.sanitize()
        # new collection 1
        collection_1 = TargetTextCollection(self._target_text_examples())
        collection_1.add_unique_key('targets', 'target_id')
        collection_1.combine_data_on_id(collection_2, 'target_id', ['different_key'],
                                        raise_on_overwrite=raise_on_overwrite, 
                                        check_same_ids=check_same_ids)
        for target_index, target_text in enumerate(collection_1.values()):
            for value_index, value in enumerate(target_text['different_key']):
                assert added_values[0][target_index][value_index] == value
        assert collection_1['2']['different_key'][0] == collection_2['2']['different_key'][-1]
        assert collection_1['2']['target_id'][0] == collection_2['2']['target_id'][-1]
        assert collection_1['2']['different_key'][-1] == collection_2['2']['different_key'][0]
        assert collection_1['2']['target_id'][-1] == collection_2['2']['target_id'][0]

        # Test the raise on over write and that it will over write the data
        assert collection_1['2']['different_key'][0] == 3
        different_values = [[[5], [6], [7, 8]]]
        collection_2 = TargetTextCollection(self._target_text_examples())
        collection_2.add_unique_key('targets', 'target_id')
        collection_2 = self.add_keys_and_values(collection_2, added_keys, different_values)
        if raise_on_overwrite:
            with pytest.raises(OverwriteError):
                collection_1.combine_data_on_id(collection_2, 'target_id', ['different_key'],
                                                raise_on_overwrite=raise_on_overwrite, 
                                                check_same_ids=check_same_ids)
        else:
            collection_1.combine_data_on_id(collection_2, 'target_id', ['different_key'],
                                            raise_on_overwrite=raise_on_overwrite, 
                                            check_same_ids=check_same_ids)
            assert collection_1['2']['different_key'] == [7, 8]
            assert collection_1['another_id']['different_key'] == [6]
            assert collection_1['0']['different_key'] == [5]

        # Test that if it does raise an error when the collections are not 
        # the same lengths if check_same_ids is True. As tests the roll back 
        # function
        another_collection = TargetTextCollection(self._target_text_examples())
        another_collection.add_unique_key('targets', 'target_id')
        another_collection = self.add_keys_and_values(another_collection, added_keys, 
                                                      different_values)
        del another_collection['another_id']
        collection_1 = TargetTextCollection(self._target_text_examples())
        collection_1.add_unique_key('targets', 'target_id')
        if check_same_ids:
            with pytest.raises(ValueError):
                collection_1.combine_data_on_id(another_collection, 'target_id', ['different_key'],
                                                raise_on_overwrite=raise_on_overwrite, 
                                                check_same_ids=check_same_ids)
        else:
            with pytest.raises(KeyError):
                assert 'different_key' not in collection_1['0']
                collection_1.combine_data_on_id(another_collection, 'target_id', ['different_key'],
                                                raise_on_overwrite=raise_on_overwrite, 
                                                check_same_ids=check_same_ids)
        # This tests that the rollback on storage works, as in when 
        # an error occurs half way through adding data that added data 
        # is removed and the collection is returned as it was before 
        # `combine_data_on_id` was called.
        assert 'different_key' not in collection_1['0']

        # Test that it works when the two collections are the same length but 
        # contain different unique target ids
        another_collection = TargetTextCollection(self._target_text_examples())
        another_collection.add_unique_key('targets', 'target_id')
        another_collection = self.add_keys_and_values(another_collection, added_keys, 
                                                      different_values)
        another_collection['another_id']['target_id'][0] = 'another_id::2'
        collection_1 = TargetTextCollection(self._target_text_examples())
        collection_1.add_unique_key('targets', 'target_id')
        if check_same_ids:
            with pytest.raises(ValueError):
                collection_1.combine_data_on_id(another_collection, 'target_id', ['different_key'],
                                                raise_on_overwrite=raise_on_overwrite, 
                                                check_same_ids=check_same_ids)
        else:
            with pytest.raises(ValueError):
                collection_1.combine_data_on_id(another_collection, 'target_id', ['different_key'],
                                                raise_on_overwrite=raise_on_overwrite, 
                                                check_same_ids=check_same_ids)
        assert 'different_key' not in collection_1['0']

        # Test the case where there are keys that are a list of a list where 
        # the inner list and not the outer list are in the target order. In 
        # this example we will assume that the two collections have a different 
        # target order.
        collection_1 = TargetTextCollection(self._target_text_examples())
        collection_1.add_unique_key('targets', 'target_id')
        collection_2 = TargetTextCollection(self._target_text_examples())
        collection_2.add_unique_key('targets', 'target_id')
        temp_ids = collection_2['2']['target_id']
        collection_2['2']._storage['target_id'] = [temp_ids[1], temp_ids[0]]

        pred_sentiment_values_0 = [[0], [1], [1], [1]]
        collection_2['0']['preds'] = pred_sentiment_values_0
        pred_sentiment_values_another = [[1], [0], [0], [2]]
        collection_2['another_id']['preds'] = pred_sentiment_values_another
        pred_sentiment_values_2 = [[1,3], [2,1], [1, 0], [1, 2]]
        collection_2['2']['preds'] = pred_sentiment_values_2
        collection_2.sanitize()
        collection_1.sanitize()

        collection_1.combine_data_on_id(collection_2, 'target_id', data_keys=['preds'], 
                                        raise_on_overwrite=raise_on_overwrite, 
                                        check_same_ids=check_same_ids)
        assert collection_1['0']['preds'] == pred_sentiment_values_0
        assert collection_1['another_id']['preds'] == pred_sentiment_values_another
        assert collection_1['2']['preds'] == [[3,1], [1,2], [0, 1], [2, 1]]
        # Ensure that an error is raised if the values returned for a 
        # key that is a list is greater than the number of targets.
        pred_sentiment_values_0 = [[0, 1], [1, 1], [1, 1], [1, 1]]
        collection_2['0']['long_preds'] = pred_sentiment_values_0
        pred_sentiment_values_another = [[1, 2], [0, 2], [0, 2], [2, 2]]
        collection_2['another_id']['long_preds'] = pred_sentiment_values_another
        pred_sentiment_values_2 = [[1,3, 4], [2,1, 4], [1, 0, 4], [1, 2, 4]]
        collection_2['2']['long_preds'] = pred_sentiment_values_2
        collection_2.sanitize()
        collection_1.sanitize()
        with pytest.raises(AssertionError):
            collection_1.combine_data_on_id(collection_2, 'target_id', data_keys=['long_preds'], 
                                            raise_on_overwrite=raise_on_overwrite, 
                                            check_same_ids=check_same_ids)
        # Ensure that it raises the AssertionError for data keys that are 
        # lists of lists of lists as these are not supported.
        pred_sentiment_values_0 = [[[0]], [[1]], [[1]], [[1]]]
        collection_2['0']['nested_preds'] = pred_sentiment_values_0
        pred_sentiment_values_another = [[[1, 2]], [[0, 2]], [[0, 2]], [[2, 2]]]
        collection_2['another_id']['nested_preds'] = pred_sentiment_values_another
        pred_sentiment_values_2 = [[[1,3, 4]], [[2,1, 4]], [[1, 0, 4]], [[1, 2, 4]]]
        collection_2['2']['nested_preds'] = pred_sentiment_values_2
        collection_2.sanitize()
        collection_1.sanitize()
        with pytest.raises(AssertionError):
            collection_1.combine_data_on_id(collection_2, 'target_id', data_keys=['nested_preds'], 
                                                raise_on_overwrite=raise_on_overwrite, 
                                                check_same_ids=check_same_ids)

    @pytest.mark.parametrize("sentiment_key", ('target_sentiments', 'category_sentiments'))
    @pytest.mark.parametrize("average_sentiment", (True, False))
    @pytest.mark.parametrize("text_sentiment_key", ('text_sentiment', 'sentence_sentiment'))
    def test_one_sentiment_text(self, sentiment_key: bool, 
                                average_sentiment: bool, 
                                text_sentiment_key: str):
        examples: List[TargetText] = self._target_text_examples()
        text = "The laptop case was great and cover was rubbish" 
        spans = [Span(4, 15), Span(30, 35), Span(0, 3), Span(16,19), Span(40,47)]
        targets = ["laptop case", "cover", "The", "was", "rubbish"]
        target_sentiments = [0, 1, 0, 1, 2] 
        categories = ["LAPTOP", "ANOTHER", "SOMETHING", "TEST", "DIFFER"]
        varied_sentiment_example = TargetText(text=text, text_id='6', targets=targets, 
                                              spans=spans, target_sentiments=target_sentiments,
                                              categories=categories)
        examples.append(varied_sentiment_example)

        import copy
        same_sentiment_twice = copy.deepcopy(examples[-2]._storage)
        same_sentiment_twice['text_id'] = '10'
        same_sentiment_twice['target_sentiments'] = [1, 1]
        same_sentiment_twice = TargetText(**same_sentiment_twice)
        examples.append(same_sentiment_twice)
        for example in examples:
            example['category_sentiments'] = example['target_sentiments']
        no_target_examples = TargetText(text_id='5', text='example text', targets=[], 
                                        target_sentiments=[], categories=[], 
                                        category_sentiments=[], spans=[])
        examples.append(no_target_examples)
        
        collection_1 = TargetTextCollection(examples)
        assert 6 == len(collection_1)

        collection_1.one_sentiment_text(sentiment_key, average_sentiment, text_sentiment_key)
        if average_sentiment:
            correct_answer = {'0': 0, 'another_id': 1, '2': [0,1], '5': None, 
                              '6': [0,1], '10': 1}
            for target_text in collection_1.values():
                text_id = target_text['text_id']
                answer = correct_answer.pop(text_id)
                if answer is None:
                    assert text_sentiment_key not in target_text
                elif isinstance(answer, list):
                    assert target_text[text_sentiment_key] in answer
                else:
                    assert answer == target_text[text_sentiment_key]

        else:
            correct_answer = {'0': 0, 'another_id': 1, '2': None, '5': None, 
                              '6': None, '10': 1}
            for target_text in collection_1.values():
                text_id = target_text['text_id']
                answer = correct_answer.pop(text_id)
                if answer is None:
                    assert text_sentiment_key not in target_text
                else:
                    assert answer == target_text[text_sentiment_key]
    
    def test_same_data(self):
        # Testing based on the same text
        a_1 = TargetText(text_id='1', text='something other')
        b_1 = TargetText(text_id='2', text='something other')
        b_2 = TargetText(text_id='3', text='something another')
        c_1 = TargetText(text_id='4', text='done')
        
        a = TargetTextCollection([a_1])
        a.name = 'a'
        b = TargetTextCollection([b_1, b_2])
        b.name = 'b'
        c = TargetTextCollection([c_1])
        c.name = 'c'

        assert not TargetTextCollection.same_data([a, c])
        assert [([(a_1, b_1)], ('a', 'b'))] == TargetTextCollection.same_data([a, b])
        assert [([(b_1, a_1)], ('b', 'a'))] == TargetTextCollection.same_data([b, c, a])

        # Testing based on ID
        c_2 = TargetText(text_id='1', text=None)
        c.add(c_2)
        assert [([(a_1, c_2)], ('a', 'c'))] == TargetTextCollection.same_data([a, c])

        # Testing the combination
        assert [([(b_1, a_1)], ('b', 'a')), 
                ([(a_1, c_2)], ('a', 'c'))] == TargetTextCollection.same_data([b, a, c])

        


        
                