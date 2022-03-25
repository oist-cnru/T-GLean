### Train model using the GLean interface
import os
import sys
import time
import argparse
import time

import numpy as np

from gpvrnn import GPvrnn as model

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help="configuration file for PV-RNN")
    args = parser.parse_args()  # will complain if no argument is given

    # Initialize PVRNN model
    rnn = model(config_file=args.config_file, task="training", epoch=-1) # resume training

    # Custom training loop
    bg_epochs = max(rnn.save_epochs // 10, 1)
    start_epoch = rnn.epoch
    start_time = time.time()
    loop_epochs = 0
    for i in range(start_epoch // bg_epochs, rnn.max_epochs // bg_epochs):
        loss = rnn.train(bg_epochs, greedy_train=False)
        loss_rec = loss["total_batch_reconstruction_loss"]
        loss_reg = loss["total_batch_regularization_loss"]
        loss_total = loss_rec+loss_reg
        print("Epoch %d  time %.0f / %.3f  loss_total %.7f  loss_rec %.7f  loss_reg %.7f" % ((i+1)*bg_epochs, time.time() - start_time, (time.time() - start_time)/((loop_epochs+1)*bg_epochs+1), loss_total, loss_rec, loss_reg))
        loop_epochs += 1
