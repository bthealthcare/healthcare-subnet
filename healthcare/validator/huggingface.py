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
from huggingface_hub import snapshot_download
from constants import BASE_DIR
from typing import List
from dotenv import load_dotenv
load_dotenv()

def download(repo_url) -> str:
    """
    Download the model of repo_url.

    Args:
    - repo_url (str): The link of model.

    Returns:
    - str: The path to the model on system.
    """
    try:
        local_dir = BASE_DIR + "/healthcare/models/validator"
        snapshot_download(repo_id = repo_url, local_dir = local_dir, token = os.getenv('ACCESS_TOKEN'))
        return local_dir + "/repo_url"
    except Exception as e:
        bt.logging.error(f"Error occured while downloading {repo_url} : {e}")
        return ""

def download_models(
    self,
    responses: List[str],
) -> List[str]:
    """
    Downloads models from huggingface.

    Args:
    - responses (List[str]): A list of responses from the miner. (e.g. username/repo_name)

    Returns:
    - List[str]: All the path to the model on system.

    """
    return [download(response) response in responses]