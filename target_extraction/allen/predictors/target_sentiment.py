from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

@Predictor.register('target-sentiment')
class TargetSentimentPredictor(Predictor):
    """
    Predictor for the Target Sentiment model classifiers
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        '''
        :param model: Allennlp Model to be used.
        :param dataset: The dataset reader to be used to convert input objects 
                        into model friendly inputs.
        :param langauge: Name of the Spacy model e.g. en_core_web_sm
        :param pos_tags: Whether the Spacy model requires to produce POS tags.
        :param fine_grained_tags: If True then returns XPOS else returns 
                                  UPOS (universal) tags.
        '''
        super().__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like either (minimum):
        1. ``{"text": "...", "targets": ["",""]}``
        2. ``{"text": "...", "categories": ["",""]}``

        If the model requires to know where the target is in the text through 
        character offset spans, this could be due to having to create left and 
        right contexts as in the case for the 
        :class:`target_extraction.allen.models.target_sentiment.split_contexts.SplitContextsClassifier`
        then the following JSON is expected:
        1. ``{"text": "...", "targets": ["",""], "spans": [[0,1], [20,22]]}``
        """
        return self._dataset_reader.text_to_instance(**json_dict)
        
        