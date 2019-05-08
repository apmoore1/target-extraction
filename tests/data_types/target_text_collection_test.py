from typing import List

import pytest

from target_extraction.data_types import TargetTextCollection, TargetText, Span

class TestTargetTextCollection:

    def _target_text_examples(self) -> List[TargetText]:
        text = 'The laptop case was great and cover was rubbish'
        text_ids = ['0', 'another_id', '2']
        spans = [[Span(4, 15)], [Span(30, 35)], [Span(4, 15), Span(30, 35)]]
        sentiments = [[0], [1], [0, 1]]
        targets = [['laptop case'], ['cover'], ['laptop case', 'cover']]
        categories = [['LAPTOP#CASE'], ['LAPTOP'], ['LAPTOP#CASE', 'LAPTOP']]

        target_text_examples = []
        for i in range(3):
            example = TargetText(text, text_ids[i], targets=targets[i],
                                 spans=spans[i], sentiments=sentiments[i],
                                 categories=categories[i])
            target_text_examples.append(example)
        return target_text_examples

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

        assert example_2['0'] == TargetText('can be any text as long as id is correct', '0')
        
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

        example_instance['sentiments'] = [0]
        assert example_instance['sentiments'] is not None
        assert example_collection['2']['sentiments'] is None

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
        


