import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import wandb
import pandas as pd
import seaborn as sns


def get_relative_confusion_matrix(confusion_matrices):
    # confusion_matrix_relative = confusion_matrix / np.sum(confusion_matrix, axis=1)
    #confusion_matrices - 1. actons, 2. color, 3.object
    confusion_matrices_relative = []
    for confusion_matrix in confusion_matrices:
        matrix = np.zeros_like(confusion_matrix)
        for i in range(confusion_matrix.shape[0]):
            matrix[i] = confusion_matrix[i] * 100 / np.sum(confusion_matrix[i]) if np.sum(
                confusion_matrix[i]) > 0 else 0
        confusion_matrices_relative.append(matrix)

    return confusion_matrices_relative


def create_confusion_matrix_plt(plot_matrix, title, save_path, floating):

    if 'action' in title:
        dictionary = ["put down", "picked up", "pushed left", "pushed right"]
    if 'color' in title:
        dictionary = ["red", "green", "blue", "yellow", "white", "brown"]
    if 'object' in title:
        dictionary = ["apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring"]
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

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    fig.tight_layout()
    fig.set_size_inches(24, 18)
    fig.tight_layout()


    return plt

def create_log_bar_chart_plt(data, title):
    labels, color = [], 'blue'
    if 'action' in title:
        labels = ["1st", "2nd", "3rd", "4th"]
    elif 'color' in title:
        labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
    elif 'object' in title:
        labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th"]

    fig, ax = plt.subplots()


    ax.bar(labels, data)
    ax.set_xlabel('Position of Actual')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title + " - Actual Predicted Ranks")

    return plt  # Returning figure instead of plt for better control


def create_line_plt(data, title):
    """
    data: list of 3 lists, each is a list of (mean, std) tuples over time
    """
    fig, ax = plt.subplots()

    labels = ["Action Accuracy", "Color Accuracy", "Object Accuracy"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    timesteps = list(range(1, len(data[0]) + 1))  # e.g., [1, 2, ..., 16]

    for i in range(3):
        # Split (mean, std) tuples
        means = [point[0] for point in data[i]]
        stds = [point[1] for point in data[i]]

        # Plot mean line
        ax.plot(timesteps, means, label=labels[i], color=colors[i])

        # Plot ±1 std deviation shaded area
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(timesteps, lower, upper, color=colors[i], alpha=0.2)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title}")
    ax.set_ylim(-5, 105)
    ax.legend()

    return plt


def visualize_correlation_matrix(correlation_matrix, title="Correlation matrix"):

    fig, ax = plt.subplots()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='BuGn',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar=True,
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    return plt





def get_evaluation(model, data_loader, device, description=""):
    dictionary = ["put down", "picked up", "pushed left", "pushed right",
                  "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
                  "red", "green", "blue", "yellow", "white", "brown"]
    actions = ["put down", "picked up", "pushed left", "pushed right"]
    objects = ["apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring"]
    colors = ["red", "green", "blue", "yellow", "white", "brown"]

    confusion_matrix_action = np.zeros((len(actions), len(actions)))
    confusion_matrix_object = np.zeros((len(objects), len(objects)))
    confusion_matrix_color = np.zeros((len(colors), len(colors)))

    action_rank_counts = torch.zeros(4)  # 4 possible ranks for actions
    object_rank_counts = torch.zeros(9)  # 9 possible ranks for objects
    color_rank_counts = torch.zeros(6)  # 6 possible ranks for colors

    model.eval()

    wrong_predictions = []

    with torch.no_grad():
        outputs = torch.zeros((len(data_loader.dataset)), 16, 19)
        labels = torch.zeros((len(data_loader.dataset), 3))
        correct_sentences = 0

        i = 0
        for (frames_batch, joints_batch, label_batch) in tqdm(data_loader, desc=description):
            frames_batch = frames_batch.to(device=device)  # (N, L, c, w, h)
            joints_batch = joints_batch.to(device=device)  # (N, L, j)
            label_batch = label_batch.to(device=device)

            batch_size = label_batch.shape[0]
            num_classes = 19
            label_batch_one_hot = torch.zeros(batch_size, num_classes).to(device=device)
            label_batch_one_hot.scatter_(1, label_batch, 1)  # (N, 19)

            output_batch = model(frames_batch, joints_batch, label_batch_one_hot, test=True)

            outputs[i:i + data_loader.batch_size] = output_batch.to(torch.device("cpu"))
            labels[i:i + data_loader.batch_size] = label_batch.to(torch.device("cpu"))
            i += data_loader.batch_size

        _, action_outputs = outputs[:, :,0:4].max(dim=2)
        _, object_outputs = outputs[:, :, 4:13].max(dim=2)
        _, color_outputs = outputs[:, :, 13:19].max(dim=2)
        object_outputs += 4
        color_outputs += 13
        #print(action_outputs, color_outputs, object_outputs)

        for n in range(outputs.shape[0]):
            confusion_matrix_action[int(labels[n, 0].item()), (action_outputs[n, -1].item())] += 1
            confusion_matrix_color[int(labels[n, 1].item())-13, (color_outputs[n, -1].item())-13] += 1
            confusion_matrix_object[int(labels[n, 2].item())-4, (object_outputs[n, -1].item())-4] += 1

            action_correct = torch.sum(action_outputs[n, -1] == labels[n, 0])
            color_correct = torch.sum(color_outputs[n, -1] == labels[n, 1])
            object_correct = torch.sum(object_outputs[n, -1] == labels[n, 2])



            action_rank = torch.argsort(outputs[n, -1, 0:4], descending=True)  # Get sorted indices
            object_rank = torch.argsort(outputs[n, -1, 4:13], descending=True)
            color_rank = torch.argsort(outputs[n, -1, 13:19], descending=True)


            action_actual_rank = (action_rank == labels[n, 0]).nonzero(as_tuple=True)[0].item()
            object_actual_rank = (object_rank == (labels[n, 2] - 4)).nonzero(as_tuple=True)[0].item()
            color_actual_rank = (color_rank == (labels[n, 1] - 13)).nonzero(as_tuple=True)[0].item()


            action_rank_counts[action_actual_rank] += 1
            object_rank_counts[object_actual_rank] += 1
            color_rank_counts[color_actual_rank] += 1

            if action_correct and color_correct and object_correct:
                correct_sentences += 1

            if (not action_correct) or (not color_correct) or (not object_correct):

                wrong_predictions.append(f"{description} sequence_{n:04d}, "
                                         f"predicted: {dictionary[action_outputs[n, -1].item()]} {dictionary[color_outputs[n, -1].item()]} {dictionary[object_outputs[n, -1].item()]}, "
                                         f"actual:    {dictionary[int(labels[n, 0].item())]} {dictionary[int(labels[n, 1].item())]} {dictionary[int(labels[n, 2].item())]}")

    sentence_wise_accuracy = correct_sentences * 100 / len(data_loader.dataset)

    action_correct_all_timesteps = []
    color_correct_all_timesteps = []
    object_correct_all_timesteps = []

    for n in range(outputs.shape[1]):  # Loop over timesteps
        # Per-sample accuracy (1 if correct, 0 if incorrect, scaled to percentage)
        action_correct = (action_outputs[:, n] == labels[:, 0]).float() * 100
        color_correct = (color_outputs[:, n] == labels[:, 1]).float() * 100
        object_correct = (object_outputs[:, n] == labels[:, 2]).float() * 100

        # Compute mean and std using torch
        action_mean = torch.mean(action_correct)
        action_std = torch.std(action_correct, unbiased=False)

        color_mean = torch.mean(color_correct)
        color_std = torch.std(color_correct, unbiased=False)

        object_mean = torch.mean(object_correct)
        object_std = torch.std(object_correct, unbiased=False)

        # Convert to Python scalars for plotting
        action_correct_all_timesteps.append((action_mean.item(), action_std.item()))
        color_correct_all_timesteps.append((color_mean.item(), color_std.item()))
        object_correct_all_timesteps.append((object_mean.item(), object_std.item()))

    # Compute percentages
    action_rank_percentages = (action_rank_counts * 100 / len(data_loader.dataset)).tolist()
    color_rank_percentages = (color_rank_counts * 100 / len(data_loader.dataset)).tolist()
    object_rank_percentages = (object_rank_counts * 100 / len(data_loader.dataset)).tolist()


    # Track wrong prediction correlations
    wrongness_data = {
        "action": [],
        "color": [],
        "object": []
    }

    for n in range(outputs.shape[0]):

        action_wrong = -1 if action_outputs[n, -1] == labels[n, 0] else 1
        color_wrong = -1 if color_outputs[n, -1] == labels[n, 1] else 1
        object_wrong = -1 if object_outputs[n, -1] == labels[n, 2] else 1
        if action_wrong + color_wrong + object_wrong > -3:
            wrongness_data["action"].append(action_wrong)
            wrongness_data["color"].append(color_wrong)
            wrongness_data["object"].append(object_wrong)

    wrongness_df = pd.DataFrame(wrongness_data)
    wrong_correlation_matrix = wrongness_df.corr()

    print("\n=== Wrong Prediction Correlation Matrix ===")
    print(wrong_correlation_matrix)


    return ([confusion_matrix_action, confusion_matrix_color, confusion_matrix_object],
            wrong_predictions,
            sentence_wise_accuracy,
            [action_rank_percentages, color_rank_percentages, object_rank_percentages],
            [action_correct_all_timesteps, color_correct_all_timesteps, object_correct_all_timesteps],
            wrong_correlation_matrix)
