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

import bittensor as bt

from healthcare.protocol import Predict
from healthcare.validator.reward import get_rewards
from healthcare.utils.uids import get_random_uids

from constant import Constant

import torchvision.transforms as transforms
import base64
from PIL import Image
from io import BytesIO

transform = transforms.Compose([
    transforms.PILToTensor()
])

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
    # Load the image
    image = Image.open(Constant.BASE_DIR + "/healthcare/dataset/miner/images/00000001_000.png")

    # Detect the image format
    image_format = image.format

    # Convert the image to a byte array
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    img_byte = buffered.getvalue()

    # Encode bytes to a Base64 string
    img_base64 = base64.b64encode(img_byte)
    img_str = img_base64.decode()

    recommended_response = "Cardiomegaly|Emphysema"

    # The dendrite client queries the network.
    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a predict query. This simply contains a single integer.
        synapse=Predict(input_image=img_str),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        # deserialize=True,
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
