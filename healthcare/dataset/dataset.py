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

import torch
import pandas as pd
from typing import List
from constants import BASE_DIR, ALL_LABELS

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
    dataframe = pd.read_csv(BASE_DIR + '/healthcare/dataset/miner/Data_Entry.csv')

    # Filter out rows where the file does not exist
    dataframe['file_exists'] = dataframe['image_name'].apply(lambda x: os.path.exists(image_dir + x))
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