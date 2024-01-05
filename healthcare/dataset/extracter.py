import tarfile
import os

import bittensor as bt
from constant import Constant

# List of your .tar.gz files
tar_files = ['images_03.tar.gz']

# Directory where you want to extract files
parent_dir = Constant.BASE_DIR + '/healthcare/dataset'
extract_to_dir = parent_dir + "/miner"

# Extract each tar.gz file
for tar_file in tar_files:
    bt.logging.info(f"extracting ... : {tar_file}")
    with tarfile.open(parent_dir + '/miner/' + tar_file, 'r:gz') as tar:
        tar.extractall(path=extract_to_dir)