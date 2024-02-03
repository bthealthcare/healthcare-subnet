import tarfile
import os

import bittensor as bt
from constant import Constant

# List of your .tar.gz files
tar_files = ['images.tar.gz']

# Directory where you want to extract files
parent_dir = Constant.BASE_DIR + '/healthcare/dataset/'
extract_to_dir = parent_dir + "validator"
os.makedirs(extract_to_dir, exist_ok=True)

# Extract each tar.gz file
for tar_file in tar_files:
    bt.logging.info(f"extracting ... : {tar_file}")
    with tarfile.open(parent_dir + tar_file, 'r:gz') as tar:
        tar.extractall(path=extract_to_dir)
