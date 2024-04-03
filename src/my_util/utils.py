import os
import shutil
from collections import namedtuple

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard import program


Bar = namedtuple("Bar", ("mean", "std", "tag"))
Curve = namedtuple("Curve", ("time", "mean", "std", "tag"))


def MakeCurve(mean=np.array([]), time=np.array([]), std=np.array([]), tag="Curve") -> Curve:
    """
    Parameters
    ----------
    time : np.ndarray
    mean : np.ndarray
    std : np.ndarray
    tag : str
    """
    if len(time) == 0 and len(mean) != 0:
        time = np.array([x for x, _ in enumerate(mean)])
    if len(std) == 0 and len(mean) != 0:
        std = np.zeros(len(mean))
    return Curve(time, mean, std, tag)


def copy_file(source_file, destination_dir):
    """
    Copies a file from the current location to the desired destination. Creates the destination directory if it does
    not yet exist.

    Parameters
    ----------
    source_file : str
        Current location of the file.
    destination_dir : str
        Destination location of the file.
    """

    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, os.path.basename(source_file))
    shutil.copy(source_file, destination_path)


def plot_curves(curves, save_path=None, name="", save_data=False):
    """
    Plots a list of curves in one figure.

    Parameters
    ----------
    curves : list
    save_path : str
    name : str
    save_data : bool
    """
    colors = plt.cm.get_cmap('tab10')

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if save_data:
            os.makedirs(os.path.join(save_path, "curve_data"), exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--')

    for idx, curve in enumerate(curves):
        plt.plot(curve.time, curve.mean, label=curve.tag, color=colors(idx))
        if save_path is not None and save_data:
            np.save(os.path.join(save_path, "curve_data", f"{name}_{curve.tag}_time"), curve.time)
            np.save(os.path.join(save_path, "curve_data", f"{name}_{curve.tag}_mean"), curve.mean)

    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)
    plt.tight_layout()
    plt.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{name}"))
    else:
        plt.show()
    plt.close()


def open_tensorboard(log_dir, port=None):
    """
    Starts a tensorboard log directory.
    Parameters
    ----------
    log_dir : str
    port : str
    """
    tb = program.TensorBoard()
    os.makedirs(log_dir, exist_ok=True)
    if port is not None:
        tb.configure(argv=[None, '--logdir', log_dir, '--port', port])
    else:
        tb.configure(argv=[None, '--logdir', log_dir])
    port = tb.launch()
    print(f"Tensorflow listening on {port}")


def token_f1_score(pred, true) -> float:
    """
    Compute the f1 score between two sets of tokens
    """
    pred_tokens = set(pred)
    true_tokens = set(true)

    # Calculate the number of shared tokens
    num_same = len(pred_tokens.intersection(true_tokens))

    # Calculate true positives, false positives, and false negatives
    tp = num_same
    fp = len(pred_tokens) - num_same
    fn = len(true_tokens) - num_same

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


def fetch_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
