To create the models that are stored within `./non_pos_model/model.tar.gz` and `pos_model/model.tar.gz` run the following two bash commands from the top of this project path:
``` bash
allennlp train tests/data/allen/predictors/target_tagger/config_char.json -s tests/data/allen/predictors/target_tagger/non_pos_model --include-package target_extraction
```
``` bash
allennlp train tests/data/allen/predictors/target_tagger/config_pos_char.json -s tests/data/allen/predictors/target_tagger/pos_model --include-package target_extraction
```
``` bash
allennlp train tests/data/allen/predictors/target_tagger/config_fine_pos_char.json -s tests/data/allen/predictors/target_tagger/fine_pos_model --include-package target_extraction
```