from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Token
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import SpacyTokenizer

@Predictor.register('target-tagger')
class TargetTaggerPredictor(Predictor):
    """
    Predictor for the 
    :class:`target_extraction.allen.models.target_tagger.TargetTagger` model.
    This predictor is very much based on the 
    :class:`from allennlp.predictors.sentence.SentenceTaggerPredictor`
    The main difference:
    1. The option to use either the tokenizer that is in the constructor of the 
       class or to provide the tokens within the JSON that is to be processed 
       thus allowing the flexiability of using your own custom tokenizer.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, 
                 language: str = 'en_core_web_sm', pos_tags: bool = True,
                 fine_grained_tags: bool = False) -> None:
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
        self._tokenizer = SpacyTokenizer(language=language, 
                                         pos_tags=pos_tags)
        self._pos_tags = pos_tags
        self._fine_grained_tags = fine_grained_tags

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like either:
        1. ``{"text": "..."}``
        2. ``{"text": "...", "tokens": ["..."]}``
        3. ``{"text": "...", "tokens": ["..."], "pos_tags": ["..."]}``
        The first will use the tokenizer and pos tagger within the constructor.
        The second will assume that only tokens are needed and are thus 
        provided. The last is similar to the second but POS tags are provided 
        as they are required by the classifier.
        """
        text = json_dict['text']
        if 'tokens' in json_dict:
            tokens = [Token(token) for token in json_dict['tokens']]
            input_dict = {'tokens': tokens, 'text': text}
            if 'pos_tags' in json_dict and \
               'pos_tags' in self._model.vocab._token_to_index:
                input_dict['pos_tags'] = json_dict['pos_tags']
            return self._dataset_reader.text_to_instance(**input_dict)
        # Using the tokenizer and pos tagger from the constructor
        tokens = self._tokenizer.tokenize(text)
        pos_tags = []
        for allen_token in tokens:            
            if self._pos_tags:
                if self._fine_grained_tags:
                    pos_tag = allen_token.tag_
                else:
                    pos_tag = allen_token.pos_
                pos_tags.append(pos_tag)
        if 'pos_tags' in self._model.vocab._token_to_index and self._pos_tags:
            return self._dataset_reader.text_to_instance(tokens=tokens, 
                                                         text=text,
                                                         pos_tags=pos_tags)
        else:
            return self._dataset_reader.text_to_instance(tokens=tokens, text=text)
        
        