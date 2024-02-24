#!/usr/bin/env python3
# Download the 56 zip files in Images_png in batches
import urllib.request
import tarfile
import os

import bittensor as bt
from constants import BASE_DIR
from healthcare.dataset.dataset import download_dataset

bt.logging.info(f"⏬ Downloading ...")
download_status = download_dataset()
if download_status:
    bt.logging.info(f"✅ Successfully downloaded the dataset.")