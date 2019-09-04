To create the ian model that is stored with `./ian/model.tar.gz` run the following bash command from the top of this project path:
``` bash
allennlp train tests/data/allen/predictors/target_sentiment/IAN/ian_config.jsonnet -s tests/data/allen/predictors/target_sentiment/IAN/ian --include-package target_extraction
```