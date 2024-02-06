# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 demon

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import glob
import shutil
import sys
import tarfile
from contextlib import contextmanager
import numpy as np
import pandas as pd
from typing import List
from constants import BASE_DIR, ALL_LABELS

import bittensor as bt
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from huggingface_hub import snapshot_download

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def load_dataset(
    csv_path: str,
    image_dir: str
):
    """
    Load dataset from images and csv file.

    Args:
    - csv_path (str): The path to the csv file.
    - image_dir (str): The path of the image directory.

    """
    # Load CSV file
    dataframe = pd.read_csv(csv_path)

    # Filter out rows where the file does not exist
    dataframe['file_exists'] = dataframe['image_name'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))
    dataframe = dataframe[dataframe['file_exists']]
    dataframe = dataframe.drop(columns=['file_exists'])

    # Preprocess images labels
    label_list = dataframe['label']
    image_list = dataframe['image_name']

    # Split all_labels and independent labels
    all_labels_list = set(ALL_LABELS.split('|'))
    split_labels = [set([item for item in label.split('|') if item != 'No Finding']) for label in label_list]

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels_list])
    
    # Transform each label
    binary_output = [mlb.transform([label]).tolist()[0] for label in split_labels]

    return image_list.values, binary_output, dataframe

def load_and_preprocess_image(image_path, target_size = (224, 224)):
    try:
        # Load image
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)

        # Resize the image using NumPy's resize. Note: np.resize and PIL's resize behave differently.
        img_array = np.array(image.smart_resize(img_array, target_size))

        # Normalize the image
        img_array = img_array / 255.0

        return img_array        
    except Exception as e:
        return "ERROR"

def download_dataset() -> bool:
    """
    Download the dataset.

    """
    repo_url = "cogent-demon/miner_dataset"
    
    # Download the dataset
    try:
        local_dir = os.path.join(BASE_DIR, "healthcare/dataset/miner")
        with suppress_stdout_stderr():
            snapshot_download(repo_id = repo_url, repo_type = "dataset", local_dir = local_dir)
        bt.logging.info(f"♻️  Extracting ...")
        # Extract the images.tar.gz
        extract_to_dir = BASE_DIR + '/healthcare/dataset/miner'
        # Get image files
        pattern = f"{extract_to_dir}/images*.tar.gz"
        tar_files = glob.glob(pattern)
        for tar_file in tar_files:
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=extract_to_dir)
        return True

    except Exception as e:
        bt.logging.error(f"❌ Error occured while downloading the dataset: {e}")
        return False