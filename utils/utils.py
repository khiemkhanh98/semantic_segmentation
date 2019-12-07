from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random




# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)



# Subtracts the mean images from ImageNet


# Compute the memory usage, for debugging

def numpy_metrics(y_pred, y_true, n_classes):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    """

    # Put y_pred and y_true under the same shape
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    # We use not_void in case the prediction falls in the void class of the groundtruth

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i))

    accuracy = np.sum(I) / len(y_pred)
    return I, U, accuracy
