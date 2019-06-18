from pathlib import Path

from target_extraction.data_types import TargetTextCollection
from target_extraction import pos_taggers
from target_extraction import tokenizers

data_fp = Path('./tests/data_types/data/complex_target_instance.json')
collection = TargetTextCollection.load_json(data_fp)
collection.tokenize(tokenizers.stanford())
collection.sequence_labels()
collection.pos_text(pos_taggers.stanford())
collection.to_json_file(Path('./example.json'))