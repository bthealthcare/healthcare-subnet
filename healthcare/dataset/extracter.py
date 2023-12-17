import tarfile
import os
from constant import Constant

# List of your .tar.gz files
tar_files = ['images_01.tar.gz']

# Directory where you want to extract files
parent_dir = Constant.BASE_DIR + '/healthcare/dataset'
extract_to_dir = parent_dir + "/miner"

# Extract each tar.gz file
for tar_file in tar_files:
    print(f"extracting ... : {tar_file}")
    with tarfile.open(parent_dir + '/miner/' + tar_file, 'r:gz') as tar:
        tar.extractall(path=extract_to_dir)