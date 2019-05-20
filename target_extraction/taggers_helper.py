'''
This module contains code that will help the following modules:

1. tokenizers
2. pos_taggers

Functions:

1. stanford_downloader - Downloads the specific Stanford NLP Neural Network 
   pipeline.
'''
from typing import Optional
from pathlib import Path

import stanfordnlp
from stanfordnlp.utils import resources

def stanford_downloader(lang: str, treebank: Optional[str] = None, 
                        download: bool = False) -> str:
    '''
    Downloads the Stanford NLP Neural Network pipelines that can be used 
    for the following tagging tasks:
    
    1. tokenizing
    2. Multi Word Tokens (MWT)
    3. POS tagging - Universal POS (UPOS) tags and depending on the language, 
       language specific POS tags (XPOS)
    4. Lemmatization 
    5. Dependency Parsing

    Each pipeline is trained per language and per treebank hence why the 
    language and treebank is required as arguments. When the treebank is not 
    given the default treebank is used. 
    
    If download is True then it will re-download the pipeline even if it 
    already exists, this might be useful if a new version has come avliable.

    Languages supported: 
    https://stanfordnlp.github.io/stanfordnlp/installation_download.html#human-
    languages-supported-by-stanfordnlp

    Reference paper:
    https://www.aclweb.org/anthology/K18-2016

    :param lang: Language of the Neural Network Pipeline to download.
    :param treebank: The neural network model to use based on the treebank 
                     it was trained from. If not given the default treebank 
                     will be used. To see which is the default treebank 
                     and the treebanks available for each language go to:
                     https://stanfordnlp.github.io/stanfordnlp/installation_
                     download.html#human-languages-supported-by-stanfordnlp
    :param download: If to re-download the model. 
    :returns: The treebank full name which this method has to resolve to it's 
              full name to find the model's directory.
    :raises ValueError: If the treebank does not exist for the given language.
                        Also will raise an error there is not a pipeline for 
                        the language given.
    '''
    if lang not in resources.default_treebanks:
        pipeline_langs = list(resources.default_treebanks.keys())
        raise ValueError(f'There is no pipeline for the language {lang}. '
                         'There are pipelines for the following languages:'
                         f' {pipeline_langs}')
    if treebank is None:
        treebank = resources.default_treebanks[lang]
    else:
        treebank = f'{lang}_{treebank}'
        if treebank not in resources.conll_shorthands:
            raise ValueError(f'The treebank {treebank} does not exist for '
                             f'{lang}. Here is a list of languages and '
                             'treebanks that do exist:\n'
                             f'{resources.conll_shorthands}')
    model_dir_name = f'{treebank}_models'
    model_download_dir = Path(Path.home(), 'stanfordnlp_resources', 
                              model_dir_name)
    if download:
        stanfordnlp.download(treebank, force=True)
    elif model_download_dir.exists():
        pass
    else:
        stanfordnlp.download(treebank, force=True)
    return treebank