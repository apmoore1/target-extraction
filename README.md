# Target Extraction
[![Build Status](https://travis-ci.org/apmoore1/target-extraction.svg?branch=master)](https://travis-ci.org/apmoore1/target-extraction) [![codecov](https://codecov.io/gh/apmoore1/target-extraction/branch/master/graph/badge.svg)](https://codecov.io/gh/apmoore1/target-extraction)

## Datasets that can be parsed
In all of our commands and cases we expect the raw data to be downloaded in to the following folder `../target_data/`
### SemEval 2014 Laptop and Restaurant
The training data can be found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and the test data [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/)

# Tutorials
Below we have created a number of notebooks to show how the package works and to explore some of the datasets that are commonly used.
## Loads and explore datasets
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
