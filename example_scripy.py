from pathlib import Path

from sklearn.model_selection import train_test_split

from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import semeval_2014
from target_extraction.tokenizers import spacy_tokenizer
from target_extraction.allen import AllenNLPModel

semeval_2014_dir = Path('..', 'original_target_datasets', 'semeval_2014',
                        ).resolve()
train_fp = Path(semeval_2014_dir, "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines", "Laptop_Train_v2.xml")
test_fp = Path(semeval_2014_dir, "ABSA_Gold_TestData", 'Laptops_Test_Gold.xml')

train_data = semeval_2014(train_fp, False)
test_data = semeval_2014(test_fp, False)

test_size = len(test_data)
print(f'Size of train {len(train_data)}, size of test {test_size}')

train_data = list(train_data.values())
train_data, val_data = train_test_split(train_data, test_size=test_size)
train_data = TargetTextCollection(train_data)
val_data = TargetTextCollection(val_data)

datasets = [train_data, val_data, test_data]
tokenizer = spacy_tokenizer()
sizes = []
for dataset in datasets:
    dataset.tokenize(tokenizer)
    dataset.sequence_labels()
    sizes.append(len(dataset))
print(f'Lengths {sizes[0]}, {sizes[1]}, {sizes[2]}')
save_dir = Path('.', 'models', 'glove_model')
param_file = Path('.', 'training_configs', 'Target_Extraction',
                  'General_Domain', 'Glove_LSTM_CRF.jsonnet')
model = AllenNLPModel('Glove', param_file, 'target-tagger', save_dir)

if not save_dir.exists():
    model.fit(train_data, val_data, test_data)
else:
    model.load()
import time
start_time = time.time()
val_iter = iter(val_data.values())
for val_predictions in model.predict_sequences(val_data.values()):
    relevant_val = next(val_iter)
    relevant_val['predicted_sequence_labels'] = val_predictions['sequence_labels']
print(time.time() - start_time)
another_time = time.time()
for val_predictions in model.predict_sequences(val_data.values()):
    pass
print(time.time() - another_time)
print('done')
print(val_data.exact_match_score('predicted_sequence_labels')[2])
test_iter = iter(test_data.values())
for test_pred in model.predict_sequences(test_data.values()):
    relevant_test = next(test_iter)
    relevant_test['predicted_sequence_labels'] = test_pred['sequence_labels']
print(test_data.exact_match_score('predicted_sequence_labels')[2])