import os
import healthcare
import bittensor as bt

def get_version() -> str:
    """
    Retrieves the version.

    """
    version_str = healthcare.__version__
    return version_str

def upgrade_version():
    """
    Upgrade if there is a new version available

    """
    local_version = get_version()
    bt.logging.info(f"You are using v{local_version}")
    try:
        os.system("git pull")
        remote_version = get_version()
        if local_version != remote_version:
            os.system("python3 -e pip install -e .")
            bt.logging.info(f"⏫ Upgraded to v{remote_version}")
    except Exception as e:
        bt.logging.error(f"❌ Error occured while upgrading the version : {e}")