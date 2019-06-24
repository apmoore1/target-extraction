import argparse
from pathlib import Path
import random as rand

from sklearn.model_selection import train_test_split

from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import semeval_2014, semeval_2016
from target_extraction import tokenizers, pos_taggers

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    valid_dataset_names = ['semeval_2014', 'semeval_2016']
    dataset_parsers = [semeval_2014, semeval_2016]

    parser = argparse.ArgumentParser()
    parser.add_argument("train_dataset_fp", type=parse_path,
                        help="File path to the training dataset")
    parser.add_argument("test_dataset_fp", type=parse_path,
                        help='File path to the test dataset')
    parser.add_argument("dataset_name", type=str, 
                        choices=valid_dataset_names)
    parser.add_argument("save_train_fp", type=parse_path, 
                        help='File Path to save the new training dataset to')
    parser.add_argument("save_val_fp", type=parse_path, 
                        help='File Path to save the new validation dataset to')
    parser.add_argument("save_test_fp", type=parse_path, 
                        help='File Path to save the new test dataset to')
    parser.add_argument("--conflict", action="store_true")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()
    print(args.conflict)

    dataset_name_parser = {name: parser for name, parser in 
                           zip(valid_dataset_names, dataset_parsers)}
    dataset_parser = dataset_name_parser[args.dataset_name]
    
    train_dataset: TargetTextCollection  = dataset_parser(args.train_dataset_fp, 
                                                          conflict=args.conflict)
    test_dataset: TargetTextCollection = dataset_parser(args.test_dataset_fp, 
                                                        conflict=args.conflict)
    if args.dataset_name == 'semeval_2016':
        train_dataset = train_dataset.one_sample_per_span(remove_empty=True)
        test_dataset = test_dataset.one_sample_per_span(remove_empty=True)
    print(f'Length of train and test: {len(train_dataset)}, {len(test_dataset)}')
    # Validation set size the same as test size.
    val_size = len(test_dataset)
    train_dataset = list(train_dataset.values())
    if args.random:
        random_state = rand.randint(1, 99999)
    else:
        random_state = 42
    train, val = train_test_split(train_dataset, test_size=val_size, 
                                  random_state=random_state)
    train_dataset = TargetTextCollection(train)
    val_dataset = TargetTextCollection(val)
    
    print(f'Length of train, val and test: {len(train_dataset)}, '
          f'{len(val_dataset)} {len(test_dataset)}')
    datasets = [train_dataset, val_dataset, test_dataset]
    tokenizer = tokenizers.stanford()
    pos_tagger = pos_taggers.stanford()
    for index, dataset in enumerate(datasets):
        print(index)
        dataset: TargetTextCollection
        dataset.tokenize(tokenizer)
        dataset.pos_text(pos_tagger)
        dataset.sequence_labels()
    print(f'Saving the JSON training dataset to {args.save_train_fp}')
    train_dataset.to_json_file(args.save_train_fp)
    print(f'Saving the JSON training dataset to {args.save_val_fp}')
    val_dataset.to_json_file(args.save_val_fp)
    print(f'Saving the JSON training dataset to {args.save_test_fp}')
    test_dataset.to_json_file(args.save_test_fp)
