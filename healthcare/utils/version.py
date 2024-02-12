import os
import codecs
import re
import healthcare
import bittensor as bt
from constants import BASE_DIR

def get_version() -> str:
    """
    Retrieves the version.

    """
    with codecs.open(os.path.join(BASE_DIR, 'healthcare/__init__.py'), encoding='utf-8') as init_file:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
        version = version_match.group(1)
        return version

def upgrade_version():
    """
    Upgrade if there is a new version available

    """
    local_version = get_version()
    bt.logging.info(f"You are using v{local_version}")
    try:
        os.system("git pull origin main > /dev/null 2>&1")
        remote_version = get_version()
        if local_version != remote_version:
            os.system("python3 -m pip install -e . > /dev/null 2>&1")
            bt.logging.info(f"⏫ Upgraded to v{remote_version}")
            os._exit(0)
    except Exception as e:
        bt.logging.error(f"❌ Error occured while upgrading the version : {e}")