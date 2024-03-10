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

from healthcare.protocol import Request
from healthcare.validator.reward import get_rewards
from healthcare.utils.uids import get_random_uids
from healthcare.validator.huggingface import download_models, remove_models
from healthcare.utils.version import get_version

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
    miner_axons = [self.metagraph.axons[uid] for uid in miner_uids]
    miner_hotkeys = [axon.hotkey for axon in miner_axons]
    miner_ips = [axon.ip for axon in miner_axons]

    # The dendrite client queries the network.
    self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=miner_axons,
        synapse=Request(version = get_version()),
        deserialize=True,
    )

    # Download models.
    responses = await download_models(self, uids = miner_uids, hotkeys = miner_hotkeys)

    # Adjust the scores based on downloaded models.
    model_paths = [response["local_dir"] for response in responses]
    commit_blocks = [response["block"] for response in responses]
    repo_ids = [response["repo_id"] for response in responses]
    rewards = get_rewards(self, model_paths=model_paths, uids = miner_uids, ips = miner_ips, commit_blocks = commit_blocks, repo_ids = repo_ids)

    # Remove cache
    remove_models(self)

    bt.logging.info(f"💯 Scored responses: {rewards}")
    # Update the scores based on the rewards.
    self.update_scores(rewards, miner_uids)
