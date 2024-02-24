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
from contextlib import contextmanager
import tarfile
import bittensor as bt
from huggingface_hub import snapshot_download
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

def download_dataset() -> bool:
    """
    Download the dataset.

    """
    repo_url = os.getenv('DATASET_LINK')
    # Check if DATASET_LINK is defined
    if not repo_url:
        bt.logging.error(f"❌ Please define the DATASET_LINK.")
        return False
    
    access_token = os.getenv('ACCESS_TOKEN')
    # Check if ACCESS_TOKEN is defined
    if not access_token:
        bt.logging.error("❌ Please define the ACCESS_TOKEN.")
        return False
    
    # Download the dataset
    try:
        local_dir = os.path.join(BASE_DIR, "healthcare/dataset/validator")
        with suppress_stdout_stderr():
            snapshot_download(repo_id = repo_url, repo_type = "dataset", local_dir = local_dir, use_auth_token = access_token)
        # Extract the images.tar.gz
        extract_to_dir = BASE_DIR + '/healthcare/dataset/validator'
        tar_file = "images.tar.gz"
        with tarfile.open(os.path.join(extract_to_dir, tar_file), 'r:gz') as tar:
            tar.extractall(path=extract_to_dir)

        return True

    except Exception as e:
        bt.logging.error(f"❌ Error occured while downloading the dataset: {e}")
        return False