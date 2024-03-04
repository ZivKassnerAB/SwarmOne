# valid until 2024-03-02 22:23:41
valid_until = 1709418221








from datetime import datetime, timedelta

def check_package_expiration(valid_until):
    """Check if the package has expired based on the valid_until timestamp."""
    current_timestamp = int(datetime.now().timestamp())

    if current_timestamp > valid_until:
        raise Exception("You are trying to use an expired package. Please install it again.")

check_package_expiration(valid_until)

class Client:
    def __init__(self, *args, **kwargs):
        try:
            raise Exception(f"Please import Client from a specific framework:\n\n"
                            f"from swarm_one.hugging_face import Client\n" 
                            f"from swarm_one.pytorch import Client\n" 
                            f"from swarm_one.tensorflow import Client\n")
        except Exception as e:
            print(e)

import logging
logging.getLogger('tensorflow').disabled = True

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def install_sourcedefender():
    print('Enabling encryption...')
    output = subprocess.run(['pip', 'install', 'sourcedefender==10.0.13'], stderr=subprocess.PIPE)

    if output.returncode != 0:
        STDOUT_RED_COLOR = '\033[91m'
        STDOUT_RESET_COLOR = '\033[0m'
        print('Encrypter installation failed, returning')
        print(STDOUT_RED_COLOR + output.stderr.decode('ASCII') + STDOUT_RESET_COLOR)
    else:
        print('Encryption enabled')


import subprocess
try:
    import sourcedefender
except ModuleNotFoundError:
    install_sourcedefender()
    import sourcedefender
