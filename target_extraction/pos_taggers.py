'''
This modules contains a set of functions that return pos tagger functions 
which can be defined by the following typing: 
Callable[[str], Tuple[List[str], List[str]]]. 
All of the functions take exactly no positional arguments but can take 
keyword arguments.

All of the functions take in a String and perform tokenisation and POS tagging 
at the same time and return both as a List of Strings where the first List 
are the tokens and the second the POS tags.

Functions:

1. stanford -- Returns both UPOS and XPOS tags where UPOS is the default. 
   Stanford Neural Network POS tagger. Tagger has the option to have been 
   trained on different languages and treebanks.
2. spacy -- Returns both UPOS and XPOS tags where UPOS is the default. 
   Spacy Neural Network POS tagger. Tagger has the option to have been trained 
   on different languages.
'''
from typing import List, Callable, Optional, Tuple
from pathlib import Path

import stanfordnlp
from stanfordnlp.utils import resources

from target_extraction.taggers_helper import stanford_downloader, spacy_downloader

def spacy_tagger(fine: bool = False, spacy_model_name: str = 'en_core_web_sm'
                 ) -> Callable[[str], Tuple[List[str], List[str]]]:
    '''
    Spacy Neural Network POS tagger which returns both UPOS and XPOS tags.

    Choice of two different POS tags:
    1. UPOS - Universal POS tags, coarse grained POS tags.
    2. XPOS - Target language fine grained POS tags.

    The XPOS for English I think is Penn Treebank set.

    If the whitespace between two words is more than one token then the Spacy 
    tagger tags it as a space, however we remove these tags.

    Languages supported: 
    https://spacy.io/usage/models

    :param fine: If True then returns XPOS else returns UPOS tags.
    :param spacy_model_name: Name of the Spacy model e.g. en_core_web_sm
    :returns: A callable that takes a String and returns the tokens and  
              associated POS tags for that String.
    '''
    spacy_model = spacy_downloader(spacy_model_name, pos_tags=True, 
                                   parse=False, ner=False)
    def _spacy_tagger(text: str
                      ) -> Callable[[str],  Tuple[List[str], List[str]]]:
        if text.strip() == '':
            return [], []
        doc = spacy_model(text)
        pos_tokens = []
        tokens = []
        for token in doc:
            if token.is_space:
                continue
            if fine:
                pos_tokens.append(token.tag_)
            else:
                pos_tokens.append(token.pos_)
            tokens.append(token.text)
        return tokens, pos_tokens
    return _spacy_tagger
    

def stanford(fine: bool = False, lang: str = 'en', 
             treebank: Optional[str] = None, 
             download: bool = False
             ) -> Callable[[str], Tuple[List[str], List[str]]]:
    '''
    Stanford Neural Network (NN) tagger that uses a highway BiLSTM that has as 
    input: 1. Word2Vec and FastText embeddings, 2. Trainable Word Vector, and 
    3. Uni-Directional LSTM over character embeddings. The UPOS predicted tag 
    is used as a feature to predict the XPOS tag within the NN.

    Choice of two different POS tags:
    1. UPOS - Universal POS tags, coarse grained POS tags.
    2. XPOS - Target language fine grained POS tags.

    The XPOS for English I think is Penn Treebank set.

    ASSUMPTIONS: The returned callable pos tagger will assume that all text 
    that is given to it, is one sentence, as this method performs sentence 
    splitting but we assume each text is one sentence and we ignore the 
    sentence splitting.

    Languages supported: 
    https://stanfordnlp.github.io/stanfordnlp/installation_download.html#human-
    languages-supported-by-stanfordnlp

    Reference paper:
    https://www.aclweb.org/anthology/K18-2016

    :param fine: If True then returns XPOS else returns UPOS tags.
    :param lang: Language of the Neural Network tokeniser
    :param treebank: The neural network model to use based on the treebank 
                     it was trained from. If not given the default treebank 
                     will be used. To see which is the default treebank 
                     and the treebanks available for each language go to:
                     https://stanfordnlp.github.io/stanfordnlp/installation_
                     download.html#human-languages-supported-by-stanfordnlp
    :param download: If to re-download the model. 
    :returns: A callable that takes a String and returns the tokens and  
              associated POS tags for that String.
    '''
    full_treebank_name = stanford_downloader(lang, treebank, download)
    nlp = stanfordnlp.Pipeline(lang=lang, processors='tokenize,mwt,pos', 
                               treebank=full_treebank_name)

    def _stanford_doc_to_text(text: str
                              ) -> Callable[[str],  Tuple[List[str], List[str]]]:
        '''
        This returns all of the pos tags in each sentence however in the 
        documentation you do have the option to use the tokens instead but 
        the words are used for downstream application hence why the words 
        were chosen over the tokens. See here for more details:
        https://stanfordnlp.github.io/stanfordnlp
        /pipeline.html#accessing-word-information
        '''
        if text.strip() == '':
            return [], []
        doc = nlp(text)
        sentences = doc.sentences
        pos_tokens = []
        tokens = []
        for sentence in sentences:
            for word in sentence.words:
                if fine:
                    pos_tokens.append(word.xpos)
                else:
                    pos_tokens.append(word.upos)
                tokens.append(word.text)
        return tokens, pos_tokens
    return _stanford_doc_to_text
