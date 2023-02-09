import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm


def make_confusion_matrix(
    cf,
    group_names=["True_Negative", "False_positive", "False_Negative", "True Positive"],
    categories=["Non_Duplicate", "Duplicate"],
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=(6, 6),
    cmap="Pastel1",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        ax=ax,
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
        linecolor="black",
        linewidths=2,
        annot_kws={"size": 17},
    )

    if xyplotlabels:
        plt.ylabel("True label", fontsize=16)
        plt.xlabel("Predicted label" + stats_text, fontsize=16)
    else:
        plt.xlabel(stats_text, fontsize=16)

    if title:
        plt.set_title(title)

    return fig


def precision(y_true, y_pred):
    """Function to calucate precision.

    Args:
        y_true (list): True labels
        y_pred (list): prediction probabilities.

    Returns:
        float: precision score.
    """
    y_pred = (y_pred > 0.5) * 1
    return "precision", precision_score(y_true, y_pred), True


def recall(y_true, y_pred):
    """Function to calucate recall.

    Args:
        y_true (list): True labels
        y_pred (list): prediction probabilities.

    Returns:
        float: recall score.
    """
    y_pred = (y_pred > 0.5) * 1
    return "recall", recall_score(y_true, y_pred), True


def f1(y_true, y_pred):
    """Function to calucate f1 score.

    Args:
        y_true (list): True labels
        y_pred (list): prediction probabilities.

    Returns:
        float: f1 score.
    """
    y_pred = (y_pred > 0.5) * 1
    return "f1", f1_score(y_true, y_pred), True


def Metrics_df(True_labels, Predictions, thres_range=[0, 1, 0.05]):
    """Calculate metrics based on threshold range.

    Args:
        True_labels (list): True labels
        Predictions (list): probability from prediction.
        thres_range (list, optional): Threshold range. Defaults to [0,1,0.05].

    Returns:
        dataframe: dataframe of metrics for each threshold.
    """
    precision = []
    recall = []
    f1 = []
    accuracy = []
    fpr = []
    thresholds = np.arange(thres_range[0], thres_range[1], thres_range[2])
    for thresh in tqdm(thresholds):
        # calculate predictions for a given threshold
        temp_pred = np.where(Predictions > thresh, 1, 0)
        precision.append(precision_score(True_labels, temp_pred))
        recall.append(recall_score(True_labels, temp_pred))
        f1.append(f1_score(True_labels, temp_pred))
        accuracy.append(accuracy_score(True_labels, temp_pred))
        fpr.append(1 - recall_score(True_labels, temp_pred, pos_label=0))
    df_metrics = pd.DataFrame(
        {
            "thresholds": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr": recall,
            "fpr": fpr,
        }
    )
    return df_metrics


def print_metrics(Y_true, pred, threshold=0.5):
    """Print a beautiful confusion matrix of prediction.

    Args:
        Y_true (list): True labels.
        pred (list): prediction probability.
        threshold (float, optional): threshold range. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    print(roc_auc_score(Y_true, pred))
    pred_1 = (pred > threshold) * 1
    cm = confusion_matrix(Y_true, pred_1)
    labels = ["True_Negative", "False_positive", "False_Negative", "True Positive"]
    categories = ["Non_Duplicate", "Duplicate"]
    fig = make_confusion_matrix(
        cm,
        figsize=(7, 7),
        group_names=labels,
        categories=categories,
        cbar=False,
        cmap="Pastel1",
    )
    return fig
