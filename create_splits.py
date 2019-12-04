import argparse
from pathlib import Path
import random as rand

from sklearn.model_selection import train_test_split

from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import semeval_2014, semeval_2016
from target_extraction.dataset_parsers import wang_2017_election_twitter_train
from target_extraction.dataset_parsers import wang_2017_election_twitter_test
from target_extraction.dataset_parsers import CACHE_DIRECTORY
from target_extraction import tokenizers, pos_taggers

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def dataset_length(task: str, dataset: TargetTextCollection) -> int:
    if task == 'extraction':
        return len(dataset)
    elif task == 'sentiment':
        return dataset.number_targets()
    return 0

if __name__ == '__main__':
    valid_dataset_names = ['semeval_2014', 'semeval_2016', 'election_twitter']
    dataset_parsers = [semeval_2014, semeval_2016, 'nothing']

    remove_errors_help = "This will remove any sequence error errors, "\
                         "else it will raise an error. This does not apply for the sentiment task"
    task_help = 'Sentiment or extraction task'

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=['extraction', 'sentiment'],
                        help=task_help)
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
    parser.add_argument("--remove_errors", action="store_true", 
                        help=remove_errors_help)
    args = parser.parse_args()
    
    if args.dataset_name == 'election_twitter':
        print(f'Downloading the Twitter dataset to {CACHE_DIRECTORY}')
        train_dataset: TargetTextCollection = wang_2017_election_twitter_train()
        test_dataset: TargetTextCollection = wang_2017_election_twitter_test()
    else:
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
    # If the task is sentiment prediction remove all of the sentences that 
    # do not have targets
    if args.task == 'sentiment':
        train_dataset = train_dataset.samples_with_targets()
        test_dataset = test_dataset.samples_with_targets()
    
    print(f'Length of train and test:')
    print([dataset_length(args.task, dataset) 
           for dataset in [train_dataset, test_dataset]])
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
    
    print(f'Length of train, val and test:')
    print([dataset_length(args.task, dataset) 
           for dataset in [train_dataset, val_dataset, test_dataset]])
    datasets = [train_dataset, val_dataset, test_dataset]
    pos_tagger = pos_taggers.spacy_tagger()
    for dataset in datasets:
        dataset: TargetTextCollection
        if args.task == 'sentiment':
            dataset.pos_text(pos_tagger)
        else:
            dataset.pos_text(pos_tagger)
            errors = dataset.sequence_labels(return_errors=True)
            if errors and not args.remove_errors:
                raise ValueError('While creating the sequence labels the '
                                f'following sequence labels have occured {errors}')
            elif errors:
                print(f'{len(errors)} number of sequence labels errors have occured'
                    ' and will be removed from the dataset')
                for error in errors:
                    del dataset[error['text_id']]
    print(f'Length of train, val and test:')
    print([dataset_length(args.task, dataset) 
           for dataset in [train_dataset, val_dataset, test_dataset]])

    args.save_train_fp.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving the JSON training dataset to {args.save_train_fp}')
    train_dataset.to_json_file(args.save_train_fp)
    print(f'Saving the JSON training dataset to {args.save_val_fp}')
    val_dataset.to_json_file(args.save_val_fp)
    print(f'Saving the JSON training dataset to {args.save_test_fp}')
    test_dataset.to_json_file(args.save_test_fp)