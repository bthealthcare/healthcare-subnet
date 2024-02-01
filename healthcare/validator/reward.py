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
from typing import List
from tensorflow.keras.models import load_model
from healthcare.dataset.dataset import load_dataset

def get_loss(model_path: str) -> float:
    """
    This method returns a loss value for the model, which is used to update the miner's score.

    Args:
    - model_path (str): The path of model.

    Returns:
    - float: The loss value for the model.
    """
    try:
        # Load dataset
        csv_path = BASE_DIR + '/healthcare/dataset/validator/Data_Entry.csv'
        image_dir = BASE_DIR + '/healthcare/dataset/validator/images'
        image_list, binary_output, dataframe = load_dataset(csv_path, image_dir)

        # Load model
        model = load_model(model_path)

        # Evaluate loss and accuracy
        loss, accuracy = model.evaluate(image_list, binary_output)
        return loss
    except Exception as e:
        bt.logging.error(f"Error occured while loading model {model_path} : {e}")
        return float('inf')

def get_rewards(
    self,
    model_paths: List[str],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given models.

    Args:
    - model_paths (List[str]): A list of path to models.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given models.
    """
    # Calculate loss of models
    loss_of_models = [[idx, get_loss(model_path)] for idx, model_path in enumerate(model_paths)]

    # Sort the list by the value, keeping track of original indices
    sorted_loss = sorted(loss_of_models, key=lambda item: item[1])
    
    # Create a dictionary to map original indices to their ranks
    rank_dict = {idx: rank for rank, (value, idx) in enumerate(sorted_data)}

    # Define groupA and groupB
    count_of_models = len(model_paths)
    top_20p_count = count_of_models / 20 + 1
    division = 0.8
    
    # Calculate reward for miners
    rewards = []
    for loss_of_model in loss_of_models:
        idx = loss_of_model[0]
        loss = loss_of_model[1]
        rank = rank_dict[idx]
        if rank < top_20p_count:
            reward = 0.9 ** rank
        else:
            reward = (0.9 ** top_20p_count) * (0.5 ** (top_20p_count - rank))
        rewards.append(reward)

    return torch.FloatTensor(rewards)
