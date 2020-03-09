# Target Extraction
[![Build Status](https://travis-ci.org/apmoore1/target-extraction.svg?branch=master)](https://travis-ci.org/apmoore1/target-extraction) [![codecov](https://codecov.io/gh/apmoore1/target-extraction/branch/master/graph/badge.svg)](https://codecov.io/gh/apmoore1/target-extraction)

## Requirements and Install
Requires Python >= 3.6.1. Been testing on Python 3.6.7 and 3.7 on Ubuntu 16.0.4 (Xenial).

Install:
1. git clone git@github.com:apmoore1/target-extraction.git
2. Go into the cloned directory and `pip install .`


## Datasets that can be parsed
In all of our commands and cases we expect the raw data to be downloaded in to the following folder `../target_data/`
### Any dataset that is the same XML format as SemEval 2014 Laptop and Restaurant
The training data for both the SemEval 2014 Laptop and Restaurant can be found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and the test data [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/). The parser for this dataset; `target_extraction.dataset_parsers.semeval_2014`

An example of the XML format expected here:
``` xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sentences>
    <sentence id="2314">
        <text>But the waiters were not great to us.</text>
        <aspectTerms>
            <aspectTerm term="waiters" polarity="negative" from="8" to="15"/>
        </aspectTerms>
        <aspectCategories>
            <aspectCategory category="service" polarity="negative"/>
        </aspectCategories>
    </sentence>
</sentences>
```

### Any dataset that is the same XML format as [SemEval 2016 task 5 subtask 1](http://alt.qcri.org/semeval2016/task5/)

An example of the XML format expected here, which contains 6 sentences of which all sentences contain aspects but only 5 sentences contain targets as `sentence id="1661043:3"` target is `NULL`:
``` xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Reviews>
    <Review rid="1661043">
        <sentences>
            <sentence id="1661043:0">
                <text>Pizza here is consistently good.</text>
                <Opinions>
                    <Opinion target="Pizza" category="FOOD#QUALITY" polarity="positive" from="0" to="5"/>
                </Opinions>
            </sentence>
            <sentence id="1661043:1">
                <text>Salads are a delicious way to begin the meal.</text>
                <Opinions>
                    <Opinion target="Salads" category="FOOD#QUALITY" polarity="positive" from="0" to="6"/>
                </Opinions>
            </sentence>
            <sentence id="1661043:2">
                <text>You should pass on the calamari.</text>
                <Opinions>
                    <Opinion target="calamari" category="FOOD#QUALITY" polarity="negative" from="23" to="31"/>
                </Opinions>
            </sentence>
            <sentence id="1661043:3">
                <text>It is thick and slightly soggy.</text>
                <Opinions>
                    <Opinion target="NULL" category="FOOD#QUALITY" polarity="negative" from="0" to="0"/>
                </Opinions>
            </sentence>
            <sentence id="1661043:4">
                <text>Decor is charming.</text>
                <Opinions>
                    <Opinion target="Decor" category="AMBIENCE#GENERAL" polarity="positive" from="0" to="5"/>
                </Opinions>
            </sentence>
            <sentence id="1661043:5">
                <text>Service is average.</text>
                <Opinions>
                    <Opinion target="Service" category="SERVICE#GENERAL" polarity="neutral" from="0" to="7"/>
                </Opinions>
            </sentence>
        </sentences>
    </Review>
</Reviews>
```

# Tutorials
Below we have created a number of notebooks to show how the package works and to explore some of the datasets that are commonly used.
## Load and explore datasets
In the following notebook we show how to load in the following two datasets that are commonly used in the literature and explore them with respect to the task of target extraction.

1. [SemEval 2014 task 4](http://alt.qcri.org/semeval2014/task4/) -- Laptop domain.
2. [SemEval 2016 task 5](http://alt.qcri.org/semeval2016/task5/) -- Restaurant domain. Training data can be found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2016-absa-restaurant-reviews-english-train-data-subtask-1/cd28e738562f11e59e2c842b2b6a04d703f9dae461bb4816a5d4320019407d23/) and Test Gold data can be found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2016-absa-restaurant-reviews-english-test-data-gold-subtask-1/42bd97c6d17511e59dbe842b2b6a04d721d1933085814d9daed8fbcbe54c0615/)

## Datasets used for Target Extraction:
1. SemeEval 2014 task 4 - Laptop 1, 2, 3, 4, 5
2. SemEval 2016 task 5 - Restaurant 1, 2, 4, 5
3. SemEval 2014 task 4 - Restaurant 3, 5
4. SemEval 2015 task 12 - Restaurant 3, 5


Papers that used those datasets numbers:
1. https://www.aclweb.org/anthology/N19-1242
2. https://www.aclweb.org/anthology/P18-2094
3. https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf
4. https://www.aclweb.org/anthology/D17-1310
5. https://www.ijcai.org/proceedings/2018/0583.pdf


From what I gather of SemEval 2014 data you can have categories and no targets but I have not seen Vice Versa. I have also not seen but I assume you can have a sentence that has neither categories nor targets. There are the following 4 sentiments, positive, negative, neutral, and conflict.
I think we want the following flags not_conflict and sentiment_to_nums

# Create JSON datasets
For Target Extraction and Target Sentiment the datasets have to be created differently due to Target Extraction using sentences that do not require target's within each sentence where as Target Sentiment does require at least one target per sentence. Therefore the JSON dataset creation has been split into whether you are creating the dataset for Extraction or Sentiment prediction.
## Target Extraction:
SemEval 2014 Laptop
``` bash
python create_splits.py extraction ../original_target_datasets/semeval_2014/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Laptop_Train_v2.xml ../original_target_datasets/semeval_2014/ABSA_Gold_TestData/Laptops_Test_Gold.xml semeval_2014 ../original_target_datasets/semeval_2014/laptop_json/train.json ../original_target_datasets/semeval_2014/laptop_json/val.json ../original_target_datasets/semeval_2014/laptop_json/test.json
```

SemEval 2014 Restaurant
``` bash
python create_splits.py extraction ../original_target_datasets/semeval_2014/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Restaurants_Train_v2.xml ../original_target_datasets/semeval_2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml semeval_2014 ../original_target_datasets/semeval_2014/restaurant_json/train.json ../original_target_datasets/semeval_2014/restaurant_json/val.json ../original_target_datasets/semeval_2014/restaurant_json/test.json
```
SemEval 2016 Restaurant
``` bash
python create_splits.py extraction ../original_target_datasets/semeval_2016/ABSA16_Restaurants_Train_SB1_v2.xml ../original_target_datasets/semeval_2016/EN_REST_SB1_TEST.xml.gold semeval_2016 ../original_target_datasets/semeval_2016/restaurant_json/train.json ../original_target_datasets/semeval_2016/restaurant_json/val.json ../original_target_datasets/semeval_2016/restaurant_json/test.json
```
## Target Sentiment Prediction
SemEval 2014 Laptop
``` bash
python create_splits.py sentiment ../original_target_datasets/semeval_2014/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Laptop_Train_v2.xml ../original_target_datasets/semeval_2014/ABSA_Gold_TestData/Laptops_Test_Gold.xml semeval_2014 ../original_target_datasets/semeval_2014/laptop_sentiment_json/train.json ../original_target_datasets/semeval_2014/laptop_sentiment_json/val.json ../original_target_datasets/semeval_2014/laptop_sentiment_json/test.json
```
SemEval 2014 Restaurant
``` bash
python create_splits.py sentiment ../original_target_datasets/semeval_2014/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Restaurants_Train_v2.xml ../original_target_datasets/semeval_2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml semeval_2014 ../original_target_datasets/semeval_2014/restaurant_sentiment_json/train.json ../original_target_datasets/semeval_2014/restaurant_sentiment_json/val.json ../original_target_datasets/semeval_2014/restaurant_sentiment_json/test.json
```

# Models
In this section we describe the different models that we have created and some standard training configurations files that create the models and replicate previous work. We break the models into Target Extraction and Sentiment Prediction models. Each of the models will be named and the names will link you to there associated README within this repository describing the model and how to run it.
## Target Extraction Models

## Sentiment Prediction Models
All of the models below work at the Target/Aspect term level.

A note on the problem of in-balanced sentiment data within the datasets noted above. As each dataset is in-balanced which can be seen through the following [tutorial](./tutorials/Load_and_Explore_Target_Extraction.ipynb) and in the table below, we can take measures to overcome this (to some extent) by weighting the loss function so that more weight is given to less frequent classes. In each of the training configurations within the model section you can add a list of weights to give to the loss function to weight the loss we therefore note below in the table the inverse frequeny weights that can be used calculated based on the following equation (reference [wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency)):

*log N/n<sub>c</sub>*

Where *N* is the size of the dataset and *n<sub>c</sub>* represents the number of samples in class *c*.

All of the Datasets below represent the training dataset statistics.

| Dataset | Num Negative (%)| Num Neutral (%)| Num Negative (%)| Dataset Size | Inv Weights |
|---|---|---|---|---|---|
| SemEval Laptop 2014 | 866 (37.44) | 460 (19.89) | 987 (42.67) | 2313 | [0.43,0.70,0.37] |
| SemEval Restaurant 2014 | 805 (22.35) | 633 (17.57) | 2164 (60.08) | 3602 | [0.65,0.76,0.22] |
| Twitter Election | 4377 (46.77) | 3615 (38.63) | 1366 (14.6) | 9358 | [0.33,0.41,0.84] |

All of the training configuration files have the SemEval 2014 Laptop dataset as the examples dataset. Also they do not have the `loss_weights` argument in the configuration but that can be added easily if you want to add that yourseleves along with any other modifications you would like to do e.g. different dataset, different word embeddings or contextualised embeddings etc.

### Target/Aspect Term
1. [TDLSTM/TCLSTM models](./training_configs/Target_Sentiment/split_contexts/README.md) from [Effective LSTMs for Target-Dependent Sentiment Classification](https://www.aclweb.org/anthology/C16-1311).

# Run allennlp
``` bash
allennlp train config_char.json -s /tmp/something --include-package target_extraction
```

allennlp train ./training_configs/Target_Sentiment/split_contexts/tdlstm.jsonnet -s /tmp/anything --include-package target_extraction

## Results
They can be found within the following [folder](./results).


## Errors
Need to remove the `{'allow_unmatched_keys': True}` from the Language Model's configuration file. You may also need to add `"vocab_namespace": "token_characters"` within the `token_characters` `embedding` value.

```
allennlp.common.checks.ConfigurationError: Extra parameters passed to BasicTextFieldEmbedder: {'allow_unmatched_keys': True}
```