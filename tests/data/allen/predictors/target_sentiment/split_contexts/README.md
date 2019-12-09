To create the models that are stored within `./tdlstm/model.tar.gz`, `./tclstm/model.tar.gz`, `./inter_tdlstm/model.tar.gz`, and `./inter_tclstm/model.tar.gz` run the following two bash commands from the top of this project path:
``` bash
allennlp train tests/data/allen/predictors/target_sentiment/split_contexts/config_tdlstm.jsonnet -s tests/data/allen/predictors/target_sentiment/split_contexts/tdlstm --include-package target_extraction
```
``` bash
allennlp train tests/data/allen/predictors/target_sentiment/split_contexts/config_tclstm.jsonnet -s tests/data/allen/predictors/target_sentiment/split_contexts/tclstm --include-package target_extraction
```
``` bash
allennlp train tests/data/allen/predictors/target_sentiment/split_contexts/config_inter_tdlstm.jsonnet -s tests/data/allen/predictors/target_sentiment/split_contexts/inter_tdlstm --include-package target_extraction
```
``` bash
allennlp train tests/data/allen/predictors/target_sentiment/split_contexts/config_inter_tclstm.jsonnet -s tests/data/allen/predictors/target_sentiment/split_contexts/inter_tclstm --include-package target_extraction
```