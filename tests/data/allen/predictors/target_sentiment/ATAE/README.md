To create the ATAE, AE, AT, and the Inter Aspect ATAE models that are stored within `./ATAE/model.tar.gz`, `./AE/model.tar.gz`, `./AT/model.tar.gz`, `./InterAspectATAE/model.tar.gz` respectively run the following bash commands from the top of this project path:
``` bash
allennlp train tests/data/allen/predictors/target_sentiment/ATAE/atae_config.jsonnet -s tests/data/allen/predictors/target_sentiment/ATAE/ATAE --include-package target_extraction
allennlp train tests/data/allen/predictors/target_sentiment/ATAE/ae_config.jsonnet -s tests/data/allen/predictors/target_sentiment/ATAE/AE --include-package target_extraction
allennlp train tests/data/allen/predictors/target_sentiment/ATAE/at_config.jsonnet -s tests/data/allen/predictors/target_sentiment/ATAE/AT --include-package target_extraction
allennlp train tests/data/allen/predictors/target_sentiment/ATAE/inter_atae_config.jsonnet -s tests/data/allen/predictors/target_sentiment/ATAE/InterAspectATAE --include-package target_extraction
```