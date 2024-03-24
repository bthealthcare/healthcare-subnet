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
import asyncio
import shutil
import sys
import bittensor as bt
import torch
from contextlib import contextmanager
from huggingface_hub import snapshot_download, HfApi
from constants import BASE_DIR
from typing import List
from healthcare.utils.chain import Chain
from dotenv import load_dotenv
load_dotenv()

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

async def download(
    self,
    uid: int,
    hotkey: str
) -> dict:
    """
    Download the miner's model from hugging face.

    Args:
    - uid (int): The uid of miner.
    - hotkey (str): The hotkey of miner.

    Returns:
    - dict: {The path of the model on the system, Block of commitment used to calculate commit time, The repo id of the model on hugging face}
    """
    local_dir = os.path.join(BASE_DIR, "healthcare/models/validator", str(uid))
    response = {"local_dir" : local_dir, "block" : float('inf'), "repo_id" : ""}
    try:
        # Retrieve miner's latest metadata from the chain.
        chain = Chain(self.config.netuid, self.subtensor, hotkey = hotkey)
        commitdata = await chain.retrieve_metadata()
        if not commitdata:
            return response
        bt.logging.info(f"⏬ Downloading the model of miner {uid} ...")

        block = commitdata["block"] # Block of the commitment
        commitment = commitdata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]
        chain_str = bytes.fromhex(hex_data).decode()

        # Get the repo_id and commit hash from the commitdata
        split_str_list = chain_str.split(" ")
        repo_id = split_str_list[0]
        commit_hash = split_str_list[1]
        
        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id)

        if commit_hash != str(repo_info.sha):
            response["block"] = float('inf')
        else:
            response["block"] = block
        response["repo_id"] = repo_id

        # Download the model
        cache_dir = os.path.join(BASE_DIR, "healthcare/models/validator/cache")
        with suppress_stdout_stderr():
            snapshot_download(repo_id = repo_id, revision = commit_hash, local_dir = local_dir, cache_dir = cache_dir)
        bt.logging.info(f"✅ Successfully downloaded the model of miner {uid}.")
    except Exception as e:
        bt.logging.error(f"❌ Error occured while downloading the model of miner {uid} : {e}")
        return response
    return response

async def download_models(
    self,
    uids: torch.LongTensor,
    hotkeys: List[str]
) -> List[str]:
    """
    Downloads models from huggingface.

    Args:
    - uids (torch.LongTensor): A list of uids of the miner.
    - hotkeys (str): A list of hotkeys of the miner.

    Returns:
    - List[str]: All the path to the model on system.

    """
    bt.logging.info(f"⏬ Downloading models ...")
    responses = [
        await download(self, uid.item(), hotkeys[idx])
        for idx, uid in enumerate(uids)
    ]
    return responses

def remove_models(
    self,
):
    """
    Remove the cache to reduce storage usage.

    """
    
    try:
        local_dir = os.path.join(BASE_DIR, "healthcare/models/validator/cache")
        shutil.rmtree(local_dir)
    except Exception as e:
        return