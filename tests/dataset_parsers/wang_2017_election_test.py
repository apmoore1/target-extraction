from pathlib import Path
from time import time
import tempfile

import pytest

from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import CACHE_DIRECTORY
from target_extraction.dataset_parsers import download_election_folder
from target_extraction.dataset_parsers import wang_2017_election_twitter_test, wang_2017_election_twitter_train

def test_download_election_folder():
    def test_files_and_folders_downloaded(dir_path: Path):
        annotation_folder = Path(dir_path, 'annotations')
        assert annotation_folder.is_dir()

        tweets_folder = Path(dir_path, 'tweets')
        assert tweets_folder.is_dir()

        train_id_fp = Path(dir_path, 'train_id.txt')
        assert train_id_fp.exists()

        test_id_fp = Path(dir_path, 'test_id.txt')
        assert test_id_fp.exists()

    # Test the normal case where it should successfully download the data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir, 'data dir')
        download_election_folder(temp_dir_path)
        test_files_and_folders_downloaded(Path(temp_dir_path, 'Wang 2017 Election Twitter'))
    # Test the case where you do not need to specify a directory it just uses 
    # the standard cache_dir = CACHE_DIRECTORY
    download_election_folder()
    test_files_and_folders_downloaded(Path(CACHE_DIRECTORY, 'Wang 2017 Election Twitter'))
        
    # Test the case where it has already been downloaded
    # Should take longer to download than to check
    with tempfile.TemporaryDirectory() as temp_dir:
        first_download_time = time()
        temp_dir_path_1 = Path(temp_dir, 'first')
        download_election_folder(temp_dir_path_1)
        first_download_time = time() - first_download_time
        test_files_and_folders_downloaded(Path(temp_dir_path_1, 'Wang 2017 Election Twitter'))
        

        second_time = time()
        download_election_folder(temp_dir_path_1)
        second_time = time() - second_time
        test_files_and_folders_downloaded(Path(temp_dir_path_1, 'Wang 2017 Election Twitter'))

        assert second_time < first_download_time
        assert second_time < 0.005
        assert first_download_time > 0.1
    
    # Test the case where only a certain number of the files have been downloaded.
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            temp_internal_dir = Path(temp_dir, 'Wang 2017 Election Twitter')
            temp_internal_dir.mkdir(parents=True, exist_ok=True)
            temp_internal_dir.touch('test_id.txt')
            download_election_folder(Path(temp_dir))

def test_train_and_test_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test both the normal cahce_dir and the given cache dir
        for data_dir in [None, Path(temp_dir, 'twitter data')]:
            train_data = wang_2017_election_twitter_train(data_dir)
            test_data = wang_2017_election_twitter_test(data_dir)
            
            assert len(train_data) > len(test_data)

            combined_data = TargetTextCollection.combine(train_data, test_data)
            assert 11899 == combined_data.number_targets()



        

        
