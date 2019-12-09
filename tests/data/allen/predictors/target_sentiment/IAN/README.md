To create the ian models that are stored within `./ian/model.tar.gz`, and `./ian_target_sequences/model.tar.gz` run the following bash commands from the top of this project path:
``` bash
allennlp train tests/data/allen/predictors/target_sentiment/IAN/ian_config.jsonnet -s tests/data/allen/predictors/target_sentiment/IAN/ian --include-package target_extraction
allennlp train tests/data/allen/predictors/target_sentiment/IAN/ian_target_sequences.jsonnet -s tests/data/allen/predictors/target_sentiment/IAN/ian_target_sequences --include-package target_extraction
allennlp train tests/data/allen/predictors/target_sentiment/IAN/inter_ian_config.jsonnet -s tests/data/allen/predictors/target_sentiment/IAN/inter_ian --include-package target_extraction
```