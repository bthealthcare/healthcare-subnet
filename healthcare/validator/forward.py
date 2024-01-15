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
import pandas as pd
import bittensor as bt

from healthcare.protocol import Predict
from healthcare.validator.reward import get_rewards
from healthcare.utils.uids import get_random_uids

from constant import Constant

import torchvision.transforms as transforms
import base64
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO

transform = transforms.Compose([
    transforms.PILToTensor()
])

def get_random_image(folder_path):
    try:
        # Get a list of all files in the folder
        files = os.listdir(folder_path)
    except Exception:
        return "", "Dataset is missing"
    
    # Filter out files that are not images
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Select a random image from the list and return its full path
    if image_files:
        random_image = random.choice(image_files)
        # Load CSV file
        csv_path = Constant.BASE_DIR + '/healthcare/dataset/validator/Data_Entry.csv'
        # Check if csv file exists
        if not os.path.exists(csv_path):
            return "", "Data entry is missing"

        try:
            dataframe = pd.read_csv(csv_path)
            # String list and corresponding image list
            string_list = dataframe['label'].tolist()
            image_list = dataframe['image_name'].tolist()

            index_of_image = image_list.index(random_image)
            image_label = string_list[index_of_image]
        except ValueError:
            return "", "Data entry doesn't match for images"

        return os.path.join(folder_path, random_image), image_label
    else:
        return "", "No images found"

def process_image(image_path):
    try:
        # Load the image
        image = Image.open(image_path)

        # Detect the image format
        image_format = image.format

        # Apply a blur effect
        # Define radius factor
        radius_factor = random.randint(0, 3)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius_factor))  # You can adjust the radius
        
        # Adjust brightness
        # Define brightness factor : >1 to increase brightness, <1 to decrease
        brightness_factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Convert the image to a byte array
        buffered = BytesIO()
        image.save(buffered, format=image_format)
        img_byte = buffered.getvalue()

        # Encode bytes to a Base64 string
        img_base64 = base64.b64encode(img_byte)
        img_str = img_base64.decode()
        return img_str
    except Exception as e:
        bt.logging.error(
            f"Failed to process the query image : {e}"
        )
        return ""


async def forward(self):
    """
    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # Define how the validator selects a miner to query, how often, etc.
    if self.step % self.config.neuron.query_time:
        return
    available_axon_size = len(self.metagraph.axons) - 1 # Except mine
    miner_selection_size = min(available_axon_size, self.config.neuron.sample_size)
    miner_uids = get_random_uids(self, k=miner_selection_size)

    # Define input_image and recommended response
    # Get the random image from the dataset
    image_path, image_label = get_random_image(Constant.BASE_DIR + "/healthcare/dataset/validator/images")
    if not image_path:
        bt.logging.error(f"Check the dataset again : {image_label}")
        self.config.neuron.disable_set_weights = True
        return
    recommended_response = image_label
    img_str = process_image(image_path)
    if not img_str:
        return
    bt.logging.info(f"recommended_response : {recommended_response}")

    # The dendrite client queries the network.
    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a predict query. This simply contains a single integer.
        synapse=Predict(input_image=img_str),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        # deserialize=True,
        timeout=30,
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    # Exit if the responses is empty
    if not responses:
        return

    # Adjust the scores based on responses from miners.
    rewards = get_rewards(self, recommended=recommended_response, responses=responses)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
