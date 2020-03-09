To create the In Context target sequences models that are stored within `./InContext/model.tar.gz`, `./Feedforward/model.tar.gz`, `./MaxPool/model.tar.gz`, `./ELMO/model.tar.gz` respectively run the following bash commands from the top of this project path:
``` bash
allennlp train tests/data/allen/predictors/target_sentiment/In_Context/in_context_config.jsonnet -s tests/data/allen/predictors/target_sentiment/In_Context/InContext --include-package target_extraction
allennlp train tests/data/allen/predictors/target_sentiment/In_Context/feedforward.jsonnet -s tests/data/allen/predictors/target_sentiment/In_Context/Feedforward --include-package target_extraction
allennlp train tests/data/allen/predictors/target_sentiment/In_Context/max_pool_config.jsonnet -s tests/data/allen/predictors/target_sentiment/In_Context/MaxPool --include-package target_extraction
allennlp train tests/data/allen/predictors/target_sentiment/In_Context/elmo_in_context_config.jsonnet -s tests/data/allen/predictors/target_sentiment/In_Context/ELMO --include-package target_extraction
```