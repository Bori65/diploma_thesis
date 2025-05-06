import argparse
import random
import numpy
from reservoirpy.nodes import Reservoir, Ridge, ScikitLearnNode, Input
from sklearn.linear_model import RidgeClassifier
from reservoirpy.mat_gen import normal

import wandb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import MultimodalSimulation
from model import VisionModel
from evaluation import get_evaluation, get_relative_confusion_matrix, create_confusion_matrix_plt


def train(visual_model, train_loader, val_loader, device, hidden_dim, leaking_rate, spectral_radius,
          input_scaling, input_connectivity, rc_connectivity, reg_parameter):

    visual_model.eval()

    # N : batch size, L : sequence length
    # frames.shape -> (N, L, 3, 224, 398)
    # joints.shape -> (N, L, 6)
    # label.shape  -> (N, 3) : one hot encoded word tokens
    # all have dtype=torch.float32


    win = normal(dtype=numpy.float32)
    source = Input()
    reservoir = Reservoir(hidden_dim, lr=leaking_rate, sr=spectral_radius, input_scaling=input_scaling,
                          Win=win, input_connectivity=input_connectivity, rc_connectivity=rc_connectivity)

    print(reservoir.hypers)
    readout_action = Ridge(ridge=reg_parameter, output_dim=4)  # Total output = 4 + 6 + 9 = 19 (action, colors, objects)
    readout_object = Ridge(ridge=reg_parameter, output_dim=9)  # Total output = 4 + 6 + 9 = 19 (action, colors, objects)
    readout_color = Ridge(ridge=reg_parameter, output_dim=6)  # Total output = 4 + 6 + 9 = 19 (action, colors, objects)


    for (frames_batch, joints_batch, label_batch) in tqdm(train_loader):
        frames_batch = frames_batch.to(device=device)  # (N, L, c, w, h)
        joints_batch = joints_batch.to(device=device)  # (N, L, j)
        label_batch = label_batch.numpy() # (N, Lout)
        output_batch = visual_model(frames_batch, joints_batch)  # shape (N, Lout, token_size) output from resnet concatenated with joints

        #ESN train
        states_train = []
        for x in output_batch:
            states = reservoir.run(x, reset=True)
            states_train.append(states[-1])

        num_actions = 4
        num_objects = 9
        num_colors = 6
        batch_size = output_batch.shape[0]

        label_batch_one_hot_action = np.zeros((batch_size, num_actions))
        label_batch_one_hot_action[np.arange(batch_size), label_batch[:, 0]] = 1

        label_batch_one_hot_object = np.zeros((batch_size, num_objects))
        label_batch_one_hot_object[np.arange(batch_size), label_batch[:, 2] - 4] = 1

        label_batch_one_hot_color = np.zeros((batch_size, num_colors))
        label_batch_one_hot_color[np.arange(batch_size), label_batch[:, 1] - 13] = 1

        states_train = np.array(states_train)

        readout_action.partial_fit(states_train, label_batch_one_hot_action)
        readout_object.partial_fit(states_train, label_batch_one_hot_object)
        readout_color.partial_fit(states_train, label_batch_one_hot_color)


    readout_action.fit()
    readout_object.fit()
    readout_color.fit()

    sentence_acc_hist = []
    action_acc_hist = []
    object_acc_hist = []
    color_acc_hist = []
    # ESN validation

    for i in [16]:
        val_confusion_matrix_absolute, final_val_wrong_predictions, final_val_sentence_wise_accuracy = get_evaluation(i, visual_model,
                                                                                                                      reservoir,
                                                                                                                      readout_action,
                                                                                                                      readout_object,
                                                                                                                      readout_color,
                                                                                                                      val_loader,
                                                                                                                      device,
                                                                                                                      "Test_on_validation_data")

        val_confusion_matrix_relative = get_relative_confusion_matrix(val_confusion_matrix_absolute)


        plt = create_confusion_matrix_plt(val_confusion_matrix_absolute,
                                          f"Final-validation-absolute-{run_name}", "./logs/", False)
        wandb.log({f"Final-validation-absolute": plt})

        plt = create_confusion_matrix_plt(val_confusion_matrix_relative,
                                          f"Final-validation-relative-{run_name}", "./logs/", True)
        wandb.log({f"Final-validation-relative": plt})


        final_val_accuracy = np.trace(val_confusion_matrix_absolute) * 100 / np.sum(val_confusion_matrix_absolute)
        final_val_action_accuracy = np.trace(val_confusion_matrix_absolute[:4, :4]) * 100 / np.sum(
            val_confusion_matrix_absolute[:4, :4])
        final_val_color_accuracy = np.trace(val_confusion_matrix_absolute[13:19, 13:19]) * 100 / np.sum(
            val_confusion_matrix_absolute[13:19, 13:19])
        final_val_object_accuracy = np.trace(val_confusion_matrix_absolute[4:13, 4:13]) * 100 / np.sum(
            val_confusion_matrix_absolute[4:13, 4:13])

        sentence_acc_hist.append(final_val_sentence_wise_accuracy)
        action_acc_hist.append(final_val_action_accuracy)
        object_acc_hist.append(final_val_object_accuracy)
        color_acc_hist.append(final_val_color_accuracy)

        wandb.log({f"Final_test_on_frames": i+1,
                   f"Final_test_sentence_wise_accuracy": final_val_sentence_wise_accuracy,
                   f"Final_test_accuracy": final_val_accuracy,
                   f"Final_test_action_accuracy": final_val_action_accuracy,
                   f"Final_test_color_accuracy": final_val_color_accuracy,
                   f"Final_test_object_accuracy": final_val_object_accuracy,

                   f"Final_test_wrong_predictions": final_val_wrong_predictions})



    #TEST
    cf_matrices_absolute = np.zeros((6, 19, 19))
    cf_matrices_absolute_gen = np.zeros((6, 19, 19))
    #for i in range(1, 7):
    for i in [1]:
        test_data = MultimodalSimulation(path=config["data_path"],
                                         visible_objects=[i],
                                         different_actions=config["different_actions"],
                                         different_colors=config["different_colors"],
                                         different_objects=config["different_objects"],
                                         exclusive_colors=config["exclusive_colors"],
                                         part="constant-test",
                                         num_samples=2000,
                                         max_frames=config["max_frames"],
                                         same_size=config["same_size"],
                                         frame_stride=config["frame_stride"],
                                         precooked=config["precooked"],
                                         feature_dim=config["convolutional_features"],
                                         transform=transform)

        gen_test_data = MultimodalSimulation(path=config["data_path"],
                                             visible_objects=[i],
                                             different_actions=config["different_actions"],
                                             different_colors=config["different_colors"],
                                             different_objects=config["different_objects"],
                                             exclusive_colors=config["exclusive_colors"],
                                             part="generalization-test",
                                             num_samples=2000,
                                             max_frames=config["max_frames"],
                                             same_size=config["same_size"],
                                             frame_stride=config["frame_stride"],
                                             precooked=config["precooked"],
                                             feature_dim=config["convolutional_features"],
                                             transform=transform)


        test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=False,
                                 num_workers=config["num_workers"])
        gen_test_loader = DataLoader(dataset=gen_test_data, batch_size=config["batch_size"], shuffle=False,
                                     num_workers=config["num_workers"])


        confusion_matrix_absolute, wrong_predictions, sentence_wise_accuracy = get_evaluation(16,visual_model,
                                                                                              reservoir,
                                                                                              readout_action,
                                                                                              readout_object,
                                                                                              readout_color,
                                                                                              test_loader,
                                                                                              config["device"],
                                                                                              f"V{i} test")

        confusion_matrix_absolute_gen, wrong_predictions_gen, sentence_wise_accuracy_gen = get_evaluation(16,visual_model,
                                                                                                          reservoir,
                                                                                                          readout_action,
                                                                                                          readout_object,
                                                                                                          readout_color,
                                                                                                          gen_test_loader,
                                                                                                          config["device"],
                                                                                                          f"V{i} generalization test")


        confusion_matrix_relative = get_relative_confusion_matrix(confusion_matrix_absolute)
        confusion_matrix_relative_gen = get_relative_confusion_matrix(confusion_matrix_absolute_gen)

        plt = create_confusion_matrix_plt(confusion_matrix_absolute,
                                          f"V{i}-test-absolute-{run_name}", "./logs/", False)
        wandb.log({f"V{i}-test-absolute": plt})

        plt = create_confusion_matrix_plt(confusion_matrix_relative,
                                          f"V{i}-test-relative-{run_name}", "./logs/", True)
        wandb.log({f"V{i}-test-relative": plt})

        plt = create_confusion_matrix_plt(confusion_matrix_absolute_gen,
                                          f"V{i}-generalization-test-absolute-{run_name}", "./logs/", False)
        wandb.log({f"V{i}-generalization-test-absolute": plt})

        plt = create_confusion_matrix_plt(confusion_matrix_relative_gen,
                                          f"V{i}-generalization-test-relative-{run_name}", "./logs/", True)
        wandb.log({f"V{i}-generalization-test-relative": plt})

        test_accuracy = np.trace(confusion_matrix_absolute) * 100 / np.sum(confusion_matrix_absolute)
        test_action_accuracy = np.trace(confusion_matrix_absolute[:4, :4]) * 100 / np.sum(
            confusion_matrix_absolute[:4, :4])
        test_color_accuracy = np.trace(confusion_matrix_absolute[13:19, 13:19]) * 100 / np.sum(
            confusion_matrix_absolute[13:19, 13:19])
        test_object_accuracy = np.trace(confusion_matrix_absolute[4:13, 4:13]) * 100 / np.sum(
            confusion_matrix_absolute[4:13, 4:13])

        gen_test_accuracy = np.trace(confusion_matrix_absolute_gen) * 100 / np.sum(confusion_matrix_absolute_gen)
        gen_test_action_accuracy = np.trace(confusion_matrix_absolute_gen[:4, :4]) * 100 / np.sum(
            confusion_matrix_absolute_gen[:4, :4])
        gen_test_color_accuracy = np.trace(confusion_matrix_absolute_gen[13:19, 13:19]) * 100 / np.sum(
            confusion_matrix_absolute_gen[13:19, 13:19])
        gen_test_object_accuracy = np.trace(confusion_matrix_absolute_gen[4:13, 4:13]) * 100 / np.sum(
            confusion_matrix_absolute_gen[4:13, 4:13])

        cf_matrices_absolute[i - 1] = confusion_matrix_absolute
        cf_matrices_absolute_gen[i - 1] = confusion_matrix_absolute_gen

        wandb.log({f"V{i}-test_sentence_wise_accuracy": sentence_wise_accuracy,
                   f"V{i}-test_accuracy": test_accuracy,
                   f"V{i}-test_action_accuracy": test_action_accuracy,
                   f"V{i}-test_color_accuracy": test_color_accuracy,
                   f"V{i}-test_object_accuracy": test_object_accuracy,
                   f"V{i}-generalization_test_sentence_wise_accuracy": sentence_wise_accuracy_gen,
                   f"V{i}-generalization_test_accuracy": gen_test_accuracy,
                   f"V{i}-generalization_test_action_accuracy": gen_test_action_accuracy,
                   f"V{i}-generalization_test_color_accuracy": gen_test_color_accuracy,
                   f"V{i}-generalization_test_object_accuracy": gen_test_object_accuracy,
                   f"V{i}-test_wrong_predictions": wrong_predictions,
                   f"V{i}-generalization_test_wrong_predictions": wrong_predictions_gen})

    test_accuracy = np.sum(np.trace(cf_matrices_absolute, axis1=1, axis2=2)) * 100 / np.sum(cf_matrices_absolute)
    test_action_accuracy = np.sum(np.trace(cf_matrices_absolute[:, :4, :4], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute[:, :4, :4])
    test_color_accuracy = np.sum(np.trace(cf_matrices_absolute[:, 13:19, 13:19], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute[:, 13:19, 13:19])
    test_object_accuracy = np.sum(np.trace(cf_matrices_absolute[:, 4:13, 4:13], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute[:, 4:13, 4:13])

    gen_test_accuracy = np.sum(np.trace(cf_matrices_absolute_gen, axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_gen)
    gen_test_action_accuracy = np.sum(
        np.trace(cf_matrices_absolute_gen[:, :4, :4], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_gen[:, :4, :4])
    gen_test_color_accuracy = np.sum(
        np.trace(cf_matrices_absolute_gen[:, 13:19, 13:19], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_gen[:, 13:19, 13:19])
    gen_test_object_accuracy = np.sum(
        np.trace(cf_matrices_absolute_gen[:, 4:13, 4:13], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_gen[:, 4:13, 4:13])

    wandb.log({"test_accuracy": test_accuracy,
               "test_action_accuracy": test_action_accuracy,
               "test_color_accuracy": test_color_accuracy,
               "test_object_accuracy": test_object_accuracy})
    print_with_time(f"Test accuracy: {test_accuracy:8.4f}%")

    wandb.log({"generalization_test_accuracy": gen_test_accuracy,
               "generalization_test_action_accuracy": gen_test_action_accuracy,
               "generalization_test_color_accuracy": gen_test_color_accuracy,
               "generalization_test_object_accuracy": gen_test_object_accuracy})
    print_with_time(f"Generalization test accuracy: {gen_test_accuracy:8.4f}%")



def get_accuracy(model, data_loader, device, description=""):
    action_correct = 0
    color_correct = 0
    object_correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for (frames_batch, joints_batch, label_batch) in tqdm(data_loader, desc=description):
            frames_batch = frames_batch.to(device=device)  # (N, L, c, w, h)
            joints_batch = joints_batch.to(device=device)  # (N, L, j)
            label_batch = label_batch.to(device=device)  # (N, Lout)

            output_batch = model(frames_batch, joints_batch)

            _, action_output_batch = torch.max(output_batch[:, 0, :], dim=1)
            _, color_output_batch = torch.max(output_batch[:, 1, :], dim=1)
            _, object_output_batch = torch.max(output_batch[:, 2, :], dim=1)

            action_correct += torch.sum(action_output_batch == label_batch[:, 0])
            color_correct += torch.sum(color_output_batch == label_batch[:, 1])
            object_correct += torch.sum(object_output_batch == label_batch[:, 2])

            total += label_batch.shape[0]

    action_accuracy = action_correct * 100 / total
    color_accuracy = color_correct * 100 / total
    object_accuracy = object_correct * 100 / total

    accuracy = (action_accuracy + color_accuracy + object_accuracy) / 3

    return total, accuracy, action_accuracy, color_accuracy, object_accuracy


def print_with_time(string):
    print(f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')} : {str(string)}\n")


if __name__ == "__main__":
    # ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing randomness") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so results are reproducible") % 2 ** 32 - 1)

    # commandline input
    parser = argparse.ArgumentParser(description="Set configuration for training.")
    # dataset
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--configuration", default="V1-C6-O9")
    # parser.add_argument("--visible_objects", default="1 2 3 4 5 6")
    # parser.add_argument("--different_actions", type=int, default=4)
    # parser.add_argument("--different_colors", type=int, default=6)
    # parser.add_argument("--different_objects", type=int, default=9)
    # parser.add_argument("--exclusive_colors", default=False)
    parser.add_argument("--max_frames", type=int, default=16)# TODO originaly was16
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--same_size", default=True)
    parser.add_argument("--no_joints", default=False)
    parser.add_argument("--num_training_samples", type=int, default=5_000)
    parser.add_argument("--num_validation_samples", type=int, default=2_500)
    # hyperparameters
    parser.add_argument("--image_features", type=int, default=512)
    parser.add_argument("--convolutional_features", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--dropout1", type=float, default=0.0)
    parser.add_argument("--dropout2", type=float, default=0.0)
    parser.add_argument("--freeze", default=True)
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden_dim", default=2048)
    parser.add_argument("--leaking_rate", default=0.02)
    parser.add_argument("--spectral_radius", default=0.98)
    parser.add_argument("--input_scaling", default=1)
    parser.add_argument("--input_connectivity", default=0.1)
    parser.add_argument("--rc_connectivity", default=0.1)
    parser.add_argument("--reg_parameter", default=0.01)
    # model architecture
    parser.add_argument("--sequence_architecture", default="esn")
    parser.add_argument("--vision_architecture", default="resnet18")
    parser.add_argument("--precooked", default=False)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--saving_path", required=True)
    args = parser.parse_args()

    print(args)


    # set training config
    config = dict(
        # dataset
        data_path=args.data_path,
        visible_objects=[int(args.configuration[1])],
        different_actions=int(args.configuration[4]),
        different_colors=int(args.configuration[7]),
        different_objects=int(args.configuration[10]),
        exclusive_colors=len(args.configuration) == 13,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        same_size=args.same_size == "True",
        no_joints=args.no_joints == "True",
        num_training_samples=min(args.num_training_samples, 5_000),
        num_validation_samples=min(args.num_validation_samples, 2_500),
        # hyperparameters
        image_features=args.image_features,
        convolutional_features=args.convolutional_features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verbose=args.verbose == "True",
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        freeze=args.freeze == True,
        pretrained=args.pretrained == True,
        device=(torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")),
        hidden_dim=args.hidden_dim,
        leaking_rate=args.leaking_rate,
        spectral_radius=args.spectral_radius,
        input_scaling=args.input_scaling,
        input_connectivity=args.input_connectivity,
        rc_connectivity=args.rc_connectivity,
        reg_parameter=args.reg_parameter,
        # model architecture
        sequence_architecture=args.sequence_architecture,
        vision_architecture=args.vision_architecture,
        precooked=args.precooked == "True",
        model_path=args.model_path)

    # path to save the final state_dict
    version = f"{datetime.now().strftime('%Y')}-{datetime.now().strftime('%m')}-{datetime.now().strftime('%d')}-" \
              f"{datetime.now().strftime('%H')}-{datetime.now().strftime('%M')}"
    visible_objects = ""
    for visible_obj in config["visible_objects"]:
        visible_objects += str(visible_obj) + "-"
    slash = "" if args.saving_path[-1] == "/" else "/"
    run_name = f"{config['sequence_architecture']}-{config['vision_architecture']}-" \
               f"{config['hidden_dim']}-{config['image_features']}-{'precooked-' if config['precooked'] else ''}" \
               f"V{visible_objects}A{config['different_actions']}-C{config['different_colors']}-" \
               f"O{config['different_objects']}-{'X-' if config['exclusive_colors'] else ''}" \
               f"{'J-' if not config['no_joints'] else ''}{version}"
    save_file = f"{args.saving_path}{slash}{run_name}.pt"
    config["save_file"] = save_file

    dataset_mean = [0.7605, 0.7042, 0.6045]
    dataset_std = [0.1832, 0.2083, 0.2902]

    torchvision_mean = [0.485, 0.456, 0.406]
    torchvision_std = [0.229, 0.224, 0.225]

    normal_transform = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    torchvision_transform = transforms.Normalize(mean=torchvision_mean, std=torchvision_std)

    if config["pretrained"]:
        transform = torchvision_transform
    else:
        transform = normal_transform
        if config["precooked"]:
            raise ValueError("Only pretrained precooks :-)")

    # datasets
    training_data = MultimodalSimulation(path=config["data_path"],
                                         visible_objects=config["visible_objects"],
                                         different_actions=config["different_actions"],
                                         different_colors=config["different_colors"],
                                         different_objects=config["different_objects"],
                                         exclusive_colors=config["exclusive_colors"],
                                         part="training",
                                         num_samples=config["num_training_samples"],
                                         max_frames=config["max_frames"],
                                         same_size=config["same_size"],
                                         frame_stride=config["frame_stride"],
                                         precooked=config["precooked"],
                                         feature_dim=config["convolutional_features"],
                                         transform=transform)

    validation_data = MultimodalSimulation(path=config["data_path"],
                                           visible_objects=config["visible_objects"],
                                           different_actions=config["different_actions"],
                                           different_colors=config["different_colors"],
                                           different_objects=config["different_objects"],
                                           exclusive_colors=config["exclusive_colors"],
                                           part="validation",
                                           num_samples=config["num_validation_samples"],
                                           max_frames=config["max_frames"],
                                           same_size=config["same_size"],
                                           frame_stride=config["frame_stride"],
                                           precooked=config["precooked"],
                                           feature_dim=config["convolutional_features"],
                                           transform=transform)

    # dataloader
    train_loader = DataLoader(dataset=training_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"])
    val_loader = DataLoader(dataset=validation_data, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"])
    # initiate model
    print(config["vision_architecture"])
    if config["sequence_architecture"] == "esn":
        model = VisionModel(vision_architecture=config["vision_architecture"],
                            pretrained_vision=config["pretrained"],
                            dropout1=config["dropout1"],
                            dropout2=config["dropout2"],
                            image_features=config["image_features"],
                            freeze=config["freeze"],
                            precooked=config["precooked"],
                            convolutional_features=config["convolutional_features"],
                            no_joints=config["no_joints"])
    else:
        raise ValueError("wrong sequence model!")

    model.to(device=config["device"])

    # weights and biases
    with wandb.init(project="Compositional Generalization in Multimodal Language Learning", entity="barbora-vician", config=config, name=run_name):

        # access all hyperparameters through wandb.config, so logging matches execution!
        config = wandb.config

        wandb.watch(model)

        # train
        train(model, train_loader, val_loader, config.device, config.hidden_dim, config.leaking_rate, config.spectral_radius,
              config.input_scaling, config.input_connectivity, config.rc_connectivity, config.reg_parameter)


