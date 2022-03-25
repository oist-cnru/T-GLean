"""Test model using the GLean interface


"""
import os
import sys
import time
import argparse

import numpy as np

from gpvrnn import GPvrnn as model

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help="configuration file for PV-RNN")
    args = parser.parse_args()  # will complain if no argument is given

    # Initialize PVRNN model
    rnn = model(config_file=args.config_file, task="testing")

    # Prior and posterior generation
    rnn.test()
