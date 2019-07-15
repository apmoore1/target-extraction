from pathlib import Path
from time import time
import tempfile

import pytest

from target_extraction.dataset_parsers import download_election_folder

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
        test_files_and_folders_downloaded(temp_dir_path)
        
        
    # Test the case where it has already been downloaded
    # Should take longer to download than to check
    with tempfile.TemporaryDirectory() as temp_dir:
        first_download_time = time()
        temp_dir_path_1 = Path(temp_dir, 'first')
        download_election_folder(temp_dir_path_1)
        first_download_time = time() - first_download_time
        test_files_and_folders_downloaded(temp_dir_path_1)
        

        second_time = time()
        download_election_folder(temp_dir_path_1)
        second_time = time() - second_time
        test_files_and_folders_downloaded(temp_dir_path_1)

        assert second_time < first_download_time
        assert second_time < 0.001
        assert first_download_time > 0.1
    
    # Test the case that the folder exists raises an error
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileExistsError):
            download_election_folder(Path(temp_dir))
        

        

        
