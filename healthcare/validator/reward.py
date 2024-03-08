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
import random
import shutil
import sys
from datetime import datetime
from contextlib import contextmanager
import torch
import numpy as np
import bittensor as bt
import requests
from typing import List
from tensorflow.keras.models import load_model
from healthcare.dataset.dataset import load_dataset, load_and_preprocess_image
from constants import BASE_DIR


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

def get_last_commit_time(model_paths: List[str]):
    """
    This method returns the last commit time of the models.

    Args:
    - model_paths (List[str]): The path of models.

    Returns:
    - List[int]: The last commit time of the models in the form of integer.
    """
    last_commit_time = []
    for model_path in model_paths:
        try:
            api_url = f"https://huggingface.co/api/models/{model_path}"
            response = requests.get(api_url)

            if response.status_code == 200:
                model_info = response.json()
                commit_time = model_info.get("lastModified")
                
                # Parse the date string to a datetime object
                date_obj = datetime.fromisoformat(commit_time[:-1])  # Removing 'Z' at the end

                # Convert the datetime object to UNIX timestamp (integer)
                timestamp = int(date_obj.timestamp())

                last_commit_time.append(timestamp)
            else:
                last_commit_time.append(float('inf'))
        except Exception as e:
            last_commit_time.append(float('inf'))
    return last_commit_time

def get_loss(model_paths: List[str], uids: List[int]):
    """
    This method returns a loss value for the model, which is used to update the miner's score.

    Args:
    - model_paths (List[str]): The path of models.

    Returns:
    - List[int, float]: The loss value for the models.
    """
    try:
        # Load dataset
        csv_path = os.path.join(BASE_DIR, 'healthcare/dataset/validator/Data_Entry.csv')
        image_dir = os.path.join(BASE_DIR, 'healthcare/dataset/validator/images')
        image_paths, binary_output, dataframe = load_dataset(csv_path, image_dir)

        # Generate x_input and y_output
        x_input = []
        y_output = []
        for idx, image_path in enumerate(image_paths):
            img = load_and_preprocess_image(os.path.join(image_dir, image_path))
            if isinstance(img, str):
                continue
            x_input.append(img)
            y_output.append(binary_output[idx])
        bt.logging.info(f"✅ Successfully loaded dataset.")
    except Exception as e:
        bt.logging.error(f"❌ Error occured while loading dataset : {e}")
        return []

    # Load model
    loss_of_models = []
    for idx, model_path in enumerate(model_paths):
        # Check if model exists
        if not model_path:
            loss = float('inf')
        else:
            bt.logging.info(f"⚒️  Processing the model of miner {uids[idx]} ...")
            try:
                model = load_model(model_path)
                # Evaluate loss and accuracy
                with suppress_stdout_stderr():
                    loss, accuracy = model.evaluate(np.array(x_input), np.array(y_output), verbose=0)
            except Exception as e:
                bt.logging.error(f"❌ Error occured while loading model : {e}")
                loss = float('inf')
        loss_of_models.append([idx, loss])
    return loss_of_models

def get_rewards(
    self,
    model_paths: List[str],
    uids: List[int],
    hug_paths: List[str]
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given models.

    Args:
    - model_paths (List[str]): A list of path to models.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given models.
    """
    bt.logging.info(f"♏ Evaluating models ...")
    # Get the last commit time of models
    last_commit_time = get_last_commit_time(hug_paths)
    latest_time = float('inf')

    # Calculate loss of models
    loss_of_models = get_loss(model_paths, uids)

    # Sort the list by the value, keeping track of original indices
    sorted_loss = sorted((value, idx) for idx, value in loss_of_models)
    
    # Create a dictionary to map original indices to their ranks
    rank_dict = {}
    current_rank = 0
    for i, (value, index) in enumerate(sorted_loss):
        if i > 0 and value != sorted_loss[i - 1][0]:
            current_rank = i
        if current_rank == 0 and last_commit_time[index] < latest_time:
            latest_time = last_commit_time[index]
        rank_dict[index] = current_rank

    # Define weight for the best miner
    weight_best_miner = 10

    # Define groupA and groupB
    group_A_rank = current_rank / 20 + 1

    alpha_A = 0.8
    alpha_B = 0.9

    # Calculate reward for miners
    rewards = []
    for loss_of_model in loss_of_models:
        idx = loss_of_model[0]
        loss = loss_of_model[1]
        rank = rank_dict[idx]
        if rank == 0 and last_commit_time[idx] == latest_time:
            reward = weight_best_miner
        elif rank < group_A_rank:
            reward = alpha_A ** rank
        else:
            reward = (alpha_A ** group_A_rank) * (alpha_B ** (rank - group_A_rank))
        rewards.append(reward)

    return torch.FloatTensor(rewards)
