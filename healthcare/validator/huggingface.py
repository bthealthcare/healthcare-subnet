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
import shutil
import sys
import bittensor as bt
from contextlib import contextmanager
from huggingface_hub import snapshot_download, HfApi
from constants import BASE_DIR
from typing import List
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

def download(self, uid, response) -> str:
    """
    Download the model of repo_url.

    Args:
    - response (List[str]): [The link of model, Token]

    Returns:
    - str: The path to the model on system.
    """
    repo_id = response[0]
    token = response[1]
    
    # Get hugging face username from the token
    try:
        api = HfApi()
        username = api.whoami(token)["name"]
    except Exception as e:
        return ""
    
    # Download the model
    try:
        repo_url = username + "/" + repo_id
        local_dir = os.path.join(BASE_DIR, "healthcare/models/validator", repo_url)
        cache_dir = os.path.join(BASE_DIR, "healthcare/models/validator/cache")
        with suppress_stdout_stderr():
            snapshot_download(repo_id = repo_url, local_dir = local_dir, token = os.getenv('ACCESS_TOKEN'), cache_dir = cache_dir)
        bt.logging.info(f"✅ Successfully downloaded the model of miner {uid}.")
        return local_dir
    except Exception as e:
        bt.logging.error(f"❌ Error occured while downloading the model of miner {uid} : {e}")
        return ""

def download_models(
    self,
    uids: List[int],
    responses: List[List[str]],
) -> List[str]:
    """
    Downloads models from huggingface.

    Args:
    - responses (List[str]): A list of responses from the miner. (e.g. username/repo_name)

    Returns:
    - List[str]: All the path to the model on system.

    """
    bt.logging.info(f"⏬ Downloading models ...")
    return [download(self, uids[idx], response) for idx, response in enumerate(responses)]

def remove_models(
    self,
):
    """
    Remove the cache to reduce storage usage.

    """
    
    try:
        local_dir = os.path.join(BASE_DIR, "healthcare/models/validator")
        shutil.rmtree(local_dir)
    except Exception as e:
        return