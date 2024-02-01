#!/usr/bin/env python3
# Download the 56 zip files in Images_png in batches
import urllib.request
import tarfile
import os

import bittensor as bt
from constants import BASE_DIR

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz'
]

# Directory where you want to extract files
parent_dir = os.path.join(BASE_DIR, 'healthcare/dataset')
extract_to_dir = parent_dir + "/miner"
os.makedirs(extract_to_dir, exist_ok=True)

for idx, link in enumerate(links):
    tar_file = 'images_%02d.tar.gz' % (idx+1)
    fn = os.path.join(extract_to_dir, 'images_%02d.tar.gz' % (idx+1))
    bt.logging.info('downloading ... : ' + tar_file)
    urllib.request.urlretrieve(link, fn)  # download the zip file

    bt.logging.info(f"extracting ... : {tar_file}")
    with tarfile.open(os.path.join(parent_dir, 'miner', tar_file), 'r:gz') as tar:
        tar.extractall(path=extract_to_dir)

bt.logging.info("Download complete. Please check the checksums")