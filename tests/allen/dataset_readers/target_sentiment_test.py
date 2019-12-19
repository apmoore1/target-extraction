from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
import pytest
import numpy as np

from target_extraction.allen.dataset_readers import TargetSentimentDatasetReader
from target_extraction.tokenizers import spacy_tokenizer

class TestTargetSentimentDatasetReader():
    @pytest.mark.parametrize("position_weights", (True, False))
    @pytest.mark.parametrize("max_position_distance", (None, 5))
    @pytest.mark.parametrize("position_embeddings", (True, False))
    @pytest.mark.parametrize("target_sequences", (True, False))
    @pytest.mark.parametrize("left_right_contexts", (True, False))
    @pytest.mark.parametrize("incl_target", (True, False))
    @pytest.mark.parametrize("reverse_right_context", (True, False))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy: bool, left_right_contexts: bool,
                            reverse_right_context: bool, incl_target: bool,
                            target_sequences: bool, position_embeddings: bool,
                            max_position_distance: int, position_weights: bool):
        # Test that a ValueError is raised if left_right_contexts is False 
        # and incl_target is True
        with pytest.raises(ValueError):
            TargetSentimentDatasetReader(lazy=lazy, incl_target=True,
                                         left_right_contexts=False, 
                                         use_categories=True)
        # Test that a ValueError is raised if left_right_contexts is False 
        # and reverse_right_context is True
        with pytest.raises(ValueError):
            TargetSentimentDatasetReader(lazy=lazy, reverse_right_context=True,
                                         left_right_contexts=False, use_categories=True)
        # Stop ValueErrors from being raised
        if reverse_right_context and not left_right_contexts:
            return
        if incl_target and not left_right_contexts:
            return
        reader = TargetSentimentDatasetReader(lazy=lazy,
                                              incl_target=incl_target,
                                              left_right_contexts=left_right_contexts,
                                              reverse_right_context=reverse_right_context,
                                              use_categories=True)
        data_dir = Path(__file__, '..', '..', '..', 'data', 'allen', 
                        'dataset_readers', 'target_sentiment').resolve()
        tokenizer = spacy_tokenizer()

        # Test the targets case and the include target case with respect to the 
        # left and right contexts
        text1 = "I charge it at night and skip taking the cord with me "\
                "because of the good battery life"
        tokens1 = tokenizer(text1)
        targets1 = ["cord", "battery life"]
        target_words1 = [tokenizer(target) for target in targets1]
        instance1 = {'text': text1, 'text words': tokens1, 
                     'targets': targets1, 'target words': target_words1, 
                     'target_sentiments': ["neutral", "positive"]}
        if left_right_contexts:
            left_texts = ["I charge it at night and skip taking the ",
                          "I charge it at night and skip taking the cord with me because of the good "]
            right_texts = [" with me because of the good battery life", ""]
            if incl_target:
                left_texts = ["I charge it at night and skip taking the cord",
                              "I charge it at night and skip taking the cord with me because of the good battery life"]
                right_texts = ["cord with me because of the good battery life", "battery life"]
            if reverse_right_context:
                right_texts = ["life battery good the of because me with", ""]
                if incl_target:
                    right_texts = ["life battery good the of because me with cord", "life battery"]
            instance1['left_contexts'] = [tokenizer(text) for text in left_texts]
            instance1['right_contexts'] = [tokenizer(text) for text in right_texts]

        text2 = "it is of high quality, has a killer GUI, is extremely stable, "\
                "is highly expandable, is bundled with lots of very good "\
                "applications, is easy to use, and is absolutely gorgeous."
        tokens2 = tokenizer(text2)
        targets2 = ["quality", "GUI", "applications", "use"]
        target_words2 = [tokenizer(target) for target in targets2]
        instance2 = {'text': text2, 'text words': tokens2, 
                    'targets': targets2, 'target words': target_words2, 
                    'target_sentiments': ["positive", "positive", "positive", "positive"]}
        
        test_target_fp = Path(data_dir, 'target_sentiments.json').resolve()
        instances = ensure_list(reader.read(str(test_target_fp)))

        assert len(instances) == 2
        true_instances = [instance1, instance2]
        for i, instance in enumerate(instances):
            # Only look at the left and right context of the first instance
            if left_right_contexts and i == 1:
                continue
            fields = instance.fields
            true_instance = true_instances[i]
            assert true_instance["text words"] == [x.text for x in fields['tokens']]
            for index, target_field in enumerate(fields['targets']):
                assert true_instance["target words"][index] == [x.text for x in target_field]
            assert true_instance['target_sentiments'] == fields['target_sentiments'].labels

            assert true_instance["text"] == fields['metadata']["text"] 
            assert true_instance["text words"] == fields['metadata']["text words"]
            assert true_instance["targets"] == fields['metadata']["targets"] 
            assert true_instance["target words"] == fields['metadata']["target words"] 
            if left_right_contexts:
                for index, left_field in enumerate(fields['left_contexts']):
                    assert true_instance["left_contexts"][index] == [x.text for x in left_field]
                for index, right_field in enumerate(fields['right_contexts']):
                    assert true_instance["right_contexts"][index] == [x.text for x in right_field]
                assert 6 == len(fields)
            else:
                assert 4 == len(fields)

        # Test the categories case
        reader = TargetSentimentDatasetReader(lazy=lazy, incl_target=False,
                                              left_right_contexts=False,
                                              use_categories=True)
        text1 = "Not only was the food outstanding, but the little perks were great."
        tokens1 = tokenizer(text1)
        instance1 = {'text': text1, 'text words': tokens1, 
                     'categories': ["food", "service"],
                     'category_sentiments': ["positive", "positive"]}

        text2 = "To be completely fair, the only redeeming factor was the food, "\
                "which was above average, but couldnt make up for all the other "\
                "deficiencies of Teodora."
        tokens2 = tokenizer(text2)
        instance2 = {'text': text2, 'text words': tokens2, 
                    'categories': ["food", "anecdotes/miscellaneous"], 
                    'category_sentiments': ["positive", "negative"]}
        
        test_category_fp = Path(data_dir, 'category_sentiments.json').resolve()
        instances = ensure_list(reader.read(str(test_category_fp)))

        assert len(instances) == 2
        true_instances = [instance1, instance2]
        for i, instance in enumerate(instances):
            fields = instance.fields
            true_instance = true_instances[i]
            assert true_instance["text words"] == [x.text for x in fields['tokens']]
            assert true_instance["categories"] == [x.text for x in fields['categories']]
            assert true_instance['category_sentiments'] == fields['category_sentiments'].labels

            assert true_instance["text"] == fields['metadata']["text"] 
            assert true_instance["text words"] == fields['metadata']["text words"]
            assert true_instance["categories"] == fields['metadata']["categories"] 
            assert 4 == len(fields)
        
        # Test the categories and target case
        reader = TargetSentimentDatasetReader(lazy=lazy, incl_target=False,
                                              left_right_contexts=left_right_contexts,
                                              reverse_right_context=reverse_right_context,
                                              use_categories=True)
        text1 = "We, there were four of us, arrived at noon - the place was "\
                "empty - and the staff acted like we were imposing on them and "\
                "they were very rude."
        tokens1 = tokenizer(text1)
        targets1 = ["staff"]
        target_words1 = [tokenizer(target) for target in targets1]
        instance1 = {'text': text1, 'text words': tokens1, 
                     'targets': targets1, 'target words': target_words1, 
                     'categories': ["SERVICE#GENERAL", "SOMETHING"],
                     'target_sentiments': ["negative"]}
        if left_right_contexts:
            left_texts = ["We, there were four of us, arrived at noon - the place was empty - and the "]
            right_texts = [" acted like we were imposing on them and they were very rude."]
            if reverse_right_context:
                right_texts = [". rude very were they and them on imposing were we like acted"]
            instance1['left_contexts'] = [tokenizer(text) for text in left_texts]
            instance1['right_contexts'] = [tokenizer(text) for text in right_texts]

        text2 = "The food was lousy - too sweet or too salty and the portions tiny."
        tokens2 = tokenizer(text2)
        targets2 = ["food", "portions"]
        target_words2 = [tokenizer(target) for target in targets2]
        instance2 = {'text': text2, 'text words': tokens2, 
                    'targets': targets2, 'target words': target_words2, 
                    'categories': ["FOOD#QUALITY", "FOOD#STYLE_OPTIONS"],
                    'target_sentiments': ["negative", "negative"]}
        
        test_target_fp = Path(data_dir, 'target_category_sentiments.json').resolve()
        instances = ensure_list(reader.read(str(test_target_fp)))

        assert len(instances) == 2
        true_instances = [instance1, instance2]
        for i, instance in enumerate(instances):
            # Only look at the left and right context of the first instance
            if left_right_contexts and i == 1:
                continue
            fields = instance.fields
            true_instance = true_instances[i]
            assert true_instance["text words"] == [x.text for x in fields['tokens']]
            for index, target_field in enumerate(fields['targets']):
                assert true_instance["target words"][index] == [x.text for x in target_field]
            assert true_instance['target_sentiments'] == fields['target_sentiments'].labels
            assert true_instance["categories"] == [x.text for x in fields['categories']]

            assert true_instance["text"] == fields['metadata']["text"] 
            assert true_instance["text words"] == fields['metadata']["text words"]
            assert true_instance["targets"] == fields['metadata']["targets"] 
            assert true_instance["target words"] == fields['metadata']["target words"] 
            assert true_instance["categories"] == fields['metadata']["categories"] 
            if left_right_contexts:
                if left_right_contexts:
                    for index, left_field in enumerate(fields['left_contexts']):
                        assert true_instance["left_contexts"][index] == [x.text for x in left_field]
                    for index, right_field in enumerate(fields['right_contexts']):
                        assert true_instance["right_contexts"][index] == [x.text for x in right_field]
                assert 7 == len(fields)
            else:
                assert 5 == len(fields)
        # Test the case for the Left right contexts case where the spans are not 
        # given
        reader = TargetSentimentDatasetReader(lazy=lazy, incl_target=False,
                                              left_right_contexts=left_right_contexts,
                                              reverse_right_context=reverse_right_context,
                                              use_categories=True)
        text_fp = Path(data_dir, 'just_text.json')
        with pytest.raises(ValueError):
            instances = ensure_list(reader.read(str(text_fp)))

        # Test the case for when we are not using the left right contexts 
        # and no targets or categories are given
        reader = TargetSentimentDatasetReader(lazy=lazy, incl_target=False,
                                              left_right_contexts=False,
                                              reverse_right_context=False,
                                              use_categories=True)
        with pytest.raises(ValueError):
            instances = ensure_list(reader.read(str(text_fp)))

        # Test the target_sequences argument
        if left_right_contexts == True:
            pass
        elif (max_position_distance is not None and 
              (not position_embeddings and not position_weights)):
            with pytest.raises(ValueError):
                reader = TargetSentimentDatasetReader(lazy=lazy, 
                                                      max_position_distance=max_position_distance,
                                                      position_embeddings=position_embeddings,
                                                      position_weights=position_weights)
        else:
            # Tests raises an error if the left_right_contexts is True
            with pytest.raises(ValueError):
                reader = TargetSentimentDatasetReader(lazy=lazy, 
                                                      left_right_contexts=True,
                                                      target_sequences=True)
            text1 = 'The laptop case was great and awfulcover'
            targets1 = ['laptop case', 'case was great']
            spans1 = [[4, 15], [11, 25]]
            if not target_sequences and not position_embeddings and not position_weights:
                text1 = "Thelaptopcasewas great and awfulcover"
                targets1 = ["laptopcase", "casewas great"]
                spans1 = [[3, 13], [9, 22]]

            tokens1 = tokenizer(text1)
            target_words1 = [tokenizer(target) for target in targets1]
            target_sequences1 = [[[0,1,0,0,0,0,0], [0,0,1,0,0,0,0]], 
                                 [[0,0,1,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0]]]
            position_embedding_seq1 = [['2','1','1','2','3','4','5'], 
                                       ['3','2','1','1','1','2','3']]
            position_weights_seq1 = [[2,1,1,2,3,4,5], [3,2,1,1,1,2,3]]
            
            
            instance1 = {'text': text1, 'text words': tokens1, 
                        'targets': targets1, 'target words': target_words1,
                        'spans': spans1, 'target_sequences': target_sequences1,
                        'target_sentiments': ["neutral", "positive"],
                        'position_weights': position_weights_seq1,
                        'position_embeddings': position_embedding_seq1}

            text2 = "it is of high quality , has a killer GUI"
            targets2 = ["quality", "GUI"]
            spans2 = [[14,21], [37,40]]
            if not target_sequences and not position_embeddings and not position_weights:
                text2 = "it is of high quality, has a killer GUI"
                spans2 = [[14, 21], [36, 39]]
            
            tokens2 = tokenizer(text2)
            target_words2 = [tokenizer(target) for target in targets2]
            target_sequences2 = [[[0,0,0,0,1,0,0,0,0,0]], [[0,0,0,0,0,0,0,0,0,1]]]
            position_embedding_seq2 = [['5','4','3','2','1','2','3','4','5','6'], 
                                       ['10','9','8','7','6','5','4','3','2','1']]
            position_weights_seq2 = [[5,4,3,2,1,2,3,4,5,6], [10,9,8,7,6,5,4,3,2,1]]
            if max_position_distance is not None:
                position_embedding_seq2 = [['5','4','3','2','1','2','3','4','5','5'], 
                                       ['5','5','5','5','5','5','4','3','2','1']]
                position_weights_seq2 = [[5,4,3,2,1,2,3,4,5,5], [5,5,5,5,5,5,4,3,2,1]]
            
            instance2 = {'text': text2, 'text words': tokens2, 
                        'targets': targets2, 'target words': target_words2,
                        'spans': spans2, 'target_sequences': target_sequences2,
                        'target_sentiments': ["positive", "positive"],
                        'position_weights': position_weights_seq2,
                        'position_embeddings': position_embedding_seq2}
            if not target_sequences:
                del instance1['target_sequences']
                del instance2['target_sequences']
            if not position_embeddings:
                del instance1['position_embeddings']
                del instance2['position_embeddings'] 
            if not position_weights:
                del instance1['position_weights']
                del instance2['position_weights']    

            reader = TargetSentimentDatasetReader(lazy=lazy, 
                                                  target_sequences=target_sequences,
                                                  position_embeddings=position_embeddings,
                                                  position_weights=position_weights,
                                                  max_position_distance=max_position_distance)
            test_target_fp = Path(data_dir, 'target_sentiment_target_sequences.json').resolve()
            instances = ensure_list(reader.read(str(test_target_fp)))

            assert len(instances) == 2
            true_instances = [instance1, instance2]
            for i, instance in enumerate(instances):
                fields = instance.fields
                true_instance = true_instances[i]
                assert true_instance["text words"] == [x.text for x in fields['tokens']]
                for index, target_field in enumerate(fields['targets']):
                    assert true_instance["target words"][index] == [x.text for x in target_field]
                assert true_instance['target_sentiments'] == fields['target_sentiments'].labels
                assert true_instance["text"] == fields['metadata']["text"] 
                assert true_instance["text words"] == fields['metadata']["text words"]
                assert true_instance["targets"] == fields['metadata']["targets"] 
                assert true_instance["target words"] == fields['metadata']["target words"] 
                number_fields = 4
                if position_embeddings:
                    number_fields += 1
                if target_sequences:
                    number_fields += 1
                if position_weights:
                    number_fields += 1
                assert number_fields == len(fields)
                if target_sequences:
                    for index, target_sequence in enumerate(fields['target_sequences']):
                        true_array = true_instance["target_sequences"][index]
                        true_array = np.array(true_array)
                        assert np.array_equal(true_array, target_sequence.array)
                if position_embeddings:
                    for index, position_embedding_field in enumerate(fields['position_embeddings']):
                        assert true_instance["position_embeddings"][index] == [x.text for x in position_embedding_field]
                if position_weights:
                    position_weight_array = np.array(true_instance["position_weights"])
                    assert np.array_equal(position_weight_array, fields['position_weights'].array)
        # Ensure raises error if the max_position_distance is less than 2
        with pytest.raises(ValueError):
            TargetSentimentDatasetReader(lazy=lazy, max_position_distance=1, 
                                         position_weights=True)

        # Test the case of both target and category sentiments.
        reader = TargetSentimentDatasetReader(lazy=lazy, incl_target=False,
                                              left_right_contexts=left_right_contexts,
                                              reverse_right_context=reverse_right_context,
                                              use_categories=True)
        text1 = "We, there were four of us, arrived at noon - the place was "\
                "empty - and the staff acted like we were imposing on them and "\
                "they were very rude."
        tokens1 = tokenizer(text1)
        targets1 = ["staff"]
        target_words1 = [tokenizer(target) for target in targets1]
        instance1 = {'text': text1, 'text words': tokens1, 
                     'targets': targets1, 'target words': target_words1, 
                     'categories': ["SERVICE#GENERAL",  "SOMETHING", "ANOTHER"],
                     'target_sentiments': ["negative"],
                     'category_sentiments': ["positive", "positive", "negative"]}
        if left_right_contexts:
            left_texts = ["We, there were four of us, arrived at noon - the place was empty - and the "]
            right_texts = [" acted like we were imposing on them and they were very rude."]
            if reverse_right_context:
                right_texts = [". rude very were they and them on imposing were we like acted"]
            instance1['left_contexts'] = [tokenizer(text) for text in left_texts]
            instance1['right_contexts'] = [tokenizer(text) for text in right_texts]

        text2 = "The food was lousy - too sweet or too salty and the portions tiny."
        tokens2 = tokenizer(text2)
        targets2 = ["food", "portions"]
        target_words2 = [tokenizer(target) for target in targets2]
        instance2 = {'text': text2, 'text words': tokens2, 
                    'targets': targets2, 'target words': target_words2, 
                    'categories': ["FOOD#QUALITY", "FOOD#STYLE_OPTIONS"],
                    'target_sentiments': ["negative", "negative"],
                    'category_sentiments': ["positive", "neutral"]}
        
        test_target_fp = Path(data_dir, 'target_sentiments_category_sentiments.json').resolve()
        instances = ensure_list(reader.read(str(test_target_fp)))

        assert len(instances) == 2
        true_instances = [instance1, instance2]
        for i, instance in enumerate(instances):
            # Only look at the left and right context of the first instance
            if left_right_contexts and i == 1:
                continue
            fields = instance.fields
            true_instance = true_instances[i]
            assert true_instance["text words"] == [x.text for x in fields['tokens']]
            for index, target_field in enumerate(fields['targets']):
                assert true_instance["target words"][index] == [x.text for x in target_field]
            assert true_instance['target_sentiments'] == fields['target_sentiments'].labels
            assert true_instance["categories"] == [x.text for x in fields['categories']]
            assert true_instance["category_sentiments"] == fields['category_sentiments'].labels
            
            assert true_instance["text"] == fields['metadata']["text"] 
            assert true_instance["text words"] == fields['metadata']["text words"]
            assert true_instance["targets"] == fields['metadata']["targets"] 
            assert true_instance["target words"] == fields['metadata']["target words"] 
            assert true_instance["categories"] == fields['metadata']["categories"] 
            if left_right_contexts:
                if left_right_contexts:
                    for index, left_field in enumerate(fields['left_contexts']):
                        assert true_instance["left_contexts"][index] == [x.text for x in left_field]
                    for index, right_field in enumerate(fields['right_contexts']):
                        assert true_instance["right_contexts"][index] == [x.text for x in right_field]
                assert 8 == len(fields)
            else:
                assert 6 == len(fields)
    
    @pytest.mark.parametrize('as_string', (True, False))
    def test_target_indicators_to_distances(self, as_string: bool):
        # one target normal case
        indicators = [[0,1,0,0]]
        test_distances = TargetSentimentDatasetReader._target_indicators_to_distances(indicators, as_string=as_string)
        true_distances = [[2,1,2,3]]
        if as_string:
            true_distances = [['2','1','2','3']]
        assert true_distances == test_distances
        # Test multiple targets with multiple target words
        indicators = [[1,1,0,0], [0,0,1,0]]
        test_distances = TargetSentimentDatasetReader._target_indicators_to_distances(indicators, as_string=as_string)
        true_distances = [[1,1,2,3], [3,2,1,2]]
        if as_string:
            true_distances = [['1','1','2','3'], ['3','2','1','2']]
        assert true_distances == test_distances
        # Test the edge case of one target one token
        indicators = [[1]]
        test_distances = TargetSentimentDatasetReader._target_indicators_to_distances(indicators, as_string=as_string)
        true_distances = [[1]]
        if as_string:
            true_distances = [['1']]
        assert true_distances == test_distances
        # Test the max distances
        indicators = [[1,1,0,0,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,0,0,0,1,1,1]]
        test_distances = TargetSentimentDatasetReader._target_indicators_to_distances(indicators, as_string=as_string, max_distance=5)
        true_distances = [[1,1,2,3,4,5,5,5], [3,2,1,1,2,3,4,5], [5,5,4,3,2,1,1,1]]
        if as_string:
            true_distances = [['1','1','2','3','4','5','5','5'], 
                              ['3','2','1','1','2','3','4','5'], 
                              ['5','5','4','3','2','1','1','1']]
        assert true_distances == test_distances
