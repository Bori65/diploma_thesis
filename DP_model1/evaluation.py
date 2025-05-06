import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import wandb
from datetime import datetime


def print_with_time(string):
    print(f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')} : {str(string)}\n")

def get_relative_confusion_matrix(confusion_matrix):
    # confusion_matrix_relative = confusion_matrix / np.sum(confusion_matrix, axis=1)
    confusion_matrix_relative = np.zeros_like(confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        confusion_matrix_relative[i] = confusion_matrix[i] * 100 / np.sum(confusion_matrix[i]) if np.sum(
            confusion_matrix[i]) > 0 else 0

    return confusion_matrix_relative


def create_confusion_matrix_plt(plot_matrix, title, save_path, floating):
    dictionary = ["put down", "picked up", "pushed left", "pushed right",
                  "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
                  "red", "green", "blue", "yellow", "white", "brown"]
    fig, ax = plt.subplots()
    im = ax.imshow(plot_matrix, vmin=0.0, vmax=np.max(plot_matrix))

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(dictionary)), labels=dictionary)
    ax.set_yticks(np.arange(len(dictionary)), labels=dictionary)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(dictionary)):
        for j in range(len(dictionary)):
            my_formatter = "{0:2.2f}" if floating else "{0:4.0f}"
            text = ax.text(j, i, f"{my_formatter.format(plot_matrix[i, j])}{'%' if floating else ''}",
                           ha="center", va="center", color="w" if plot_matrix[i, j] < np.max(plot_matrix) / 2 else "0")

    ax.set_title(title)
    # plt.figtext(0.1, 0.5, 'Tatsächlich', horizontalalignment='center', va="center", rotation=90)
    # plt.figtext(0.5, 0.01, 'Vorhersage', horizontalalignment='center')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    fig.tight_layout()
    fig.set_size_inches(24, 18)
    fig.tight_layout()
    # Path(save_path).mkdir(exist_ok=True)
    # plt.savefig(f"{save_path}{title}.png", dpi=100, bbox_inches='tight')

    return plt


def get_evaluation(i, model, reservoir, readout_action, readout_object, readout_color, data_loader, device, description=""):

    action_correct, color_correct, object_correct, sentence_correct, total = 0, 0, 0, 0, 0

    dictionary = [
        "put down", "picked up", "pushed left", "pushed right",
        "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
        "red", "green", "blue", "yellow", "white", "brown"
    ]
    confusion_matrix = np.zeros((len(dictionary), len(dictionary)))
    wrong_predictions = []

    for frames_batch, joints_batch, label_batch in tqdm(data_loader, desc=description):
        # Move data to device
        frames_batch = frames_batch.to(device)
        joints_batch = joints_batch.to(device)
        label_batch = label_batch.numpy()

        # Get model predictions
        output_batch = model(frames_batch, joints_batch)

        y_action_pred, y_object_pred, y_color_pred = [], [], []
        for x in output_batch:
            #states = reservoir.run(x[:i+1], reset=True)
            states = reservoir.run(x, reset=True)

            y_action_pred.append(readout_action.run(states[-1, np.newaxis]))
            y_object_pred.append(readout_object.run(states[-1, np.newaxis]))
            y_color_pred.append(readout_color.run(states[-1, np.newaxis]))

        # Convert predictions to classes
        y_action_pred_class = np.array([np.argmax(y) for y in y_action_pred])
        y_object_pred_class = np.array([np.argmax(y) for y in y_object_pred])
        y_color_pred_class = np.array([np.argmax(y) for y in y_color_pred])

        # Ground truth labels
        Y_action_test_class = label_batch[:, 0]
        Y_object_test_class = label_batch[:, 2] - 4
        Y_color_test_class = label_batch[:, 1] - 13

        # Update accuracy counters
        action_correct += np.sum(Y_action_test_class == y_action_pred_class)
        object_correct += np.sum(Y_object_test_class == y_object_pred_class)
        color_correct += np.sum(Y_color_test_class == y_color_pred_class)
        sentence_correct += np.sum(
            (Y_action_test_class == y_action_pred_class) &
            (Y_object_test_class == y_object_pred_class) &
            (Y_color_test_class == y_color_pred_class)
        )

        # Update confusion matrix and log wrong predictions
        for n in range(label_batch.shape[0]):
            confusion_matrix[Y_action_test_class[n], y_action_pred_class[n]] += 1
            confusion_matrix[Y_object_test_class[n] + 4, y_object_pred_class[n] + 4] += 1
            confusion_matrix[Y_color_test_class[n] + 13, y_color_pred_class[n] + 13] += 1

            if not (
                    Y_action_test_class[n] == y_action_pred_class[n] and
                    Y_object_test_class[n] == y_object_pred_class[n] and
                    Y_color_test_class[n] == y_color_pred_class[n]
            ):
                wrong_predictions.append(
                    f"{description} sequence_{n:04d}, "
                    f"predicted: {dictionary[y_action_pred_class[n]]} {dictionary[y_color_pred_class[n] + 13]} {dictionary[y_object_pred_class[n] + 4]}, "
                    f"actual: {dictionary[Y_action_test_class[n]]} {dictionary[Y_color_test_class[n] + 13]} {dictionary[Y_object_test_class[n] + 4]}"
                )

        total += label_batch.shape[0]

    # Calculate accuracies
    action_accuracy = action_correct * 100 / total
    color_accuracy = color_correct * 100 / total
    object_accuracy = object_correct * 100 / total
    sentence_accuracy = sentence_correct * 100 / total
    overall_accuracy = (action_accuracy + color_accuracy + object_accuracy) / 3

    print_with_time(f"{description} sentence accuracy: {sentence_accuracy:8.4f}%")

    return confusion_matrix, wrong_predictions, sentence_accuracy
