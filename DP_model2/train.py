import argparse
import random
from copy import deepcopy
import wandb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import MultimodalSimulation
from models import CompositModel
from evaluation import *


def training_loop(n_epochs, model, train_loader, val_loader, optimizer, loss_fn, device):
    best_validation_accuracy = -1.0
    best_model_state = None

    for epoch in range(n_epochs):
        loss_train = 0.0
        training_action_correct = 0
        training_color_correct = 0
        training_object_correct = 0
        training_total = 0

        # train
        model.train()


        # N : batch size, L : sequence length
        # frames.shape -> (N, L, 3, 224, 398)
        # joints.shape -> (N, L, 6)
        # label.shape  -> (N, 3) : one hot encoded word tokens
        # all have dtype=torch.float32


        for (frames_batch, joints_batch, label_batch) in tqdm(train_loader, desc="Training"):
            frames_batch = frames_batch.to(device=device)  # (N, L, c, w, h)
            joints_batch = joints_batch.to(device=device)  # (N, L, j)
            label_batch = label_batch.to(device=device)  # (N, Lout)

            batch_size = label_batch.shape[0]
            num_classes = 19
            label_batch_one_hot = torch.zeros(batch_size, num_classes).to(device=device)
            label_batch_one_hot.scatter_(1, label_batch, 1)  # (N, 19)

            output_batch = model(frames_batch, joints_batch, label_batch_one_hot, test=False)  # shape (N, token_size)


            #for p in model.parameters():
                #if p.grad is not None:
                    #print(p.grad.data.shape)

            #print("output shape: ", output_batch.shape)
            #print(label_batch_one_hot.shape)
            loss = loss_fn(output_batch, label_batch_one_hot)

            # backward pass
            optimizer.zero_grad()


            #before_update = {name: param.clone().detach() for name, param in model.named_parameters()}

            # Perform backward and optimizer step
            loss.backward()
            optimizer.step()

            # Compare updated and old parameters
            #for name, param in model.named_parameters():
            #    if not torch.equal(param, before_update[name]):
            #        print(f"{name} was updated.")
            #    else:
            #        print(f"{name} was NOT updated.")

            loss_train += loss.item()

            # calculate training accuracy
            _, action_output_batch = output_batch[:, 0:4].max(dim=1)
            _, object_output_batch = output_batch[:, 4:13].max(dim=1)
            _, color_output_batch = output_batch[:, 13:19].max(dim=1)

            object_output_batch += 4
            color_output_batch += 13
            #print(action_output_batch, color_output_batch, object_output_batch)
            #print(label_batch[:, 0], label_batch[:, 1], label_batch[:, 2])

            training_action_correct += torch.sum(action_output_batch == label_batch[:, 0])
            training_color_correct += torch.sum(color_output_batch == label_batch[:, 1])
            training_object_correct += torch.sum(object_output_batch == label_batch[:, 2])

            training_total += label_batch.shape[0]

        training_action_accuracy = training_action_correct * 100 / training_total
        training_color_accuracy = training_color_correct * 100 / training_total
        training_object_accuracy = training_object_correct * 100 / training_total

        training_accuracy = (training_action_accuracy + training_color_accuracy + training_object_accuracy) / 3

        epoch_loss = loss_train / len(train_loader)

        # validation
        _, validation_accuracy, validation_action_accuracy, validation_color_accuracy, validation_object_accuracy = get_accuracy(
            model, val_loader, device, "Validation")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_state = deepcopy(model.state_dict())

        # logging
        wandb.log({"epoch": epoch,
                   "loss": epoch_loss,
                   "training_accuracy": training_accuracy,
                   "training_action_accuracy": training_action_accuracy,
                   "training_color_accuracy": training_color_accuracy,
                   "training_object_accuracy": training_object_accuracy,
                   "validation_accuracy": validation_accuracy,
                   "validation_action_accuracy": validation_action_accuracy,
                   "validation_color_accuracy": validation_color_accuracy,
                   "validation_object_accuracy": validation_object_accuracy})

        print_with_time(f'Epoch: {epoch:5}, Loss: {epoch_loss:16.14f}, Training accuracy: {training_accuracy:8.4f}%, '
                        f'Validation accuracy: {validation_accuracy:8.4f}%')

    return best_model_state


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

            batch_size = label_batch.shape[0]
            num_classes = 19
            label_batch_one_hot = torch.zeros(batch_size, num_classes).to(device=device)
            label_batch_one_hot.scatter_(1, label_batch, 1)  # (N, 19)

            output_batch = model(frames_batch, joints_batch, label_batch_one_hot, test=False)

            _, action_output_batch = output_batch[:, 0:4].max(dim=1)
            _, object_output_batch = output_batch[:, 4:13].max(dim=1)
            _, color_output_batch = output_batch[:, 13:19].max(dim=1)

            object_output_batch += 4
            color_output_batch += 13

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
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--same_size", default=True)
    parser.add_argument("--no_joints", default=False)
    parser.add_argument("--num_training_samples", type=int, default=5_000)
    parser.add_argument("--num_validation_samples", type=int, default=2_500)
    # hyperparameters
    parser.add_argument("--image_features", type=int, default=256)
    parser.add_argument("--convolutional_features", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--dropout1", type=float, default=0.5)
    parser.add_argument("--dropout2", type=float, default=0.0)
    parser.add_argument("--freeze", default=False)
    parser.add_argument("--leaking_rate", default=0.02)
    parser.add_argument("--lambda_reg", default=0.01)
    parser.add_argument("--spectral_radius", default=0.98)
    parser.add_argument("--output_steps", default='last')
    parser.add_argument("--readout_training", default='gd')
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--pretrained", default="True")
    parser.add_argument("--device", default="cuda")
    # model architecture
    parser.add_argument("--sequence_architecture", default="esn_pytorch")
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
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verbose=args.verbose == "True",
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        freeze=args.freeze == "True",
        leaking_rate=args.leaking_rate,
        lambda_reg=args.lambda_reg,
        spectral_radius=args.spectral_radius,
        output_steps=args.output_steps,
        readout_training=args.readout_training,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        pretrained=args.pretrained == "True",
        device=(torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")),
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
    data_conf = f"V{visible_objects}A{config['different_actions']}-C{config['different_colors']}-" \
               f"O{config['different_objects']}-{'X-' if config['exclusive_colors'] else ''}" \
               f"{'J-' if not config['no_joints'] else ''}"

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
    if config["sequence_architecture"] == "esn_pytorch":
        model = CompositModel(vision_architecture=config["vision_architecture"],
                              pretrained_vision=config["pretrained"],
                              dropout1=config["dropout1"],
                              dropout2=config["dropout2"],
                              image_features=config["image_features"],
                              seq2seq_architecture=config["sequence_architecture"],
                              hidden_dim=config["hidden_dim"],
                              freeze=config["freeze"],
                              leaking_rate=config['leaking_rate'],
                              lambda_reg=config['lambda_reg'],
                              spectral_radius=config['spectral_radius'],
                              output_steps=config['output_steps'],
                              readout_training=config['readout_training'],
                              precooked=config["precooked"],
                              convolutional_features=config["convolutional_features"],
                              no_joints=config["no_joints"],
                              )
    else:
        raise ValueError("wrong sequence model!")

    model.to(device=config["device"])

    for name, param in model.named_parameters():
        if param.requires_grad:
            ...
            #print(name, param.data.shape)
    print(sum(p.numel() for p in model.esn_model.readout.parameters()))
    print(sum(p.numel() for p in model.vision_model.parameters()))
    # optimizer

    if config["optimizer"] == "rms_prop":
        optimizer = optim.RMSprop(list(model.vision_model.parameters()) + list(model.esn_model.readout.parameters()), lr=config["learning_rate"])
    elif config["optimizer"] == "adam":
        optimizer = optim.Adam(list(model.vision_model.parameters()) + list(model.esn_model.readout.parameters()), lr=config["learning_rate"])
    else:
        raise ValueError("Wrong optimizer!")

    # loss function
    mse_loss = nn.MSELoss()


    # weights and biases
    with (wandb.init(project="Comp gen with TorchESN", entity="barbora-vician", config=config, name=run_name)):

        # access all hyperparameters through wandb.config, so logging matches execution!
        config = wandb.config

        wandb.watch(model)

        # train
        best_model_state = training_loop(config.num_epochs, model, train_loader, val_loader, optimizer,
                                         mse_loss, config.device)

        # test with the best model only
        model.load_state_dict(best_model_state)
        model.to(device=config["device"])

        (train_confusion_matrices_absolute, final_train_wrong_predictions, final_train_sentence_wise_accuracy,
         final_train_rank_percentages, final_train_correct_all_timestaps, final_train_correlation_matrix)= get_evaluation(
            model, train_loader,
            config["device"],
            f"Final training")
        (val_confusion_matrices_absolute, final_val_wrong_predictions, final_val_sentence_wise_accuracy,
         final_val_rank_percentages, final_val_correct_all_timestaps, final_val_correlation_matrix) = get_evaluation(
            model,
            val_loader,
            config[
                "device"],
            f"Final validation")


        train_confusion_matrices_relative = get_relative_confusion_matrix(train_confusion_matrices_absolute)
        val_confusion_matrices_relative = get_relative_confusion_matrix(val_confusion_matrices_absolute)

        plt = create_line_plt(final_train_correct_all_timestaps, f"Final-training-{data_conf}")
        wandb.log({f"Final-training-over-time": wandb.Image(plt)})

        plt = create_line_plt(final_val_correct_all_timestaps,f"Final-validation-{data_conf}")
        wandb.log({f"Final-validation-over-time": wandb.Image(plt)})

        plt = visualize_correlation_matrix(final_train_correlation_matrix,f"Final training {data_conf}Correlation matrix")
        wandb.log({f"Final_training-Correlation_matrix": wandb.Image(plt)})

        plt = visualize_correlation_matrix(final_val_correlation_matrix, f"Final validation {data_conf}Correlation matrix")
        wandb.log({f"Final_validation-Correlation_matrix": wandb.Image(plt)})

        for i, sentence_part in enumerate(['action', 'color', 'object']):


            plt = create_log_bar_chart_plt(final_train_rank_percentages[i], f"Final-training-{sentence_part}")
            wandb.log({f"Final-training-actual-predicted-{sentence_part}": wandb.Image(plt)})

            plt = create_log_bar_chart_plt(final_val_rank_percentages[i], f"Final-validation-{sentence_part}")
            wandb.log({f"Final-validation-actual-predicted-{sentence_part}": wandb.Image(plt)})

            plt = create_confusion_matrix_plt(train_confusion_matrices_absolute[i],
                                              f"Final-training-absolute-{sentence_part}-{run_name}", "./logs/", False)
            wandb.log({f"Final-training-absolute-{sentence_part}": plt})

            plt = create_confusion_matrix_plt(train_confusion_matrices_relative[i],
                                              f"Final-training-relative-{sentence_part}-{run_name}", "./logs/", True)
            wandb.log({f"Final-training-relative-{sentence_part}": plt})

            plt = create_confusion_matrix_plt(val_confusion_matrices_absolute[i],
                                              f"Final-validation-absolute-{sentence_part}-{run_name}", "./logs/", False)
            wandb.log({f"Final-validation-absolute-{sentence_part}": plt})

            plt = create_confusion_matrix_plt(val_confusion_matrices_relative[i],
                                              f"Final-validation-relative-{sentence_part}-{run_name}", "./logs/", True)
            wandb.log({f"Final-validation-relative-{sentence_part}": plt})

            final_train_action_accuracy = np.trace(train_confusion_matrices_absolute[0]) * 100 / np.sum(
                train_confusion_matrices_absolute[0])
            final_train_color_accuracy = np.trace(train_confusion_matrices_absolute[1]) * 100 / np.sum(
                train_confusion_matrices_absolute[1])
            final_train_object_accuracy = np.trace(train_confusion_matrices_absolute[2]) * 100 / np.sum(
                train_confusion_matrices_absolute[2])

            final_val_action_accuracy = np.trace(val_confusion_matrices_absolute[0]) * 100 / np.sum(
                val_confusion_matrices_absolute[0])
            final_val_color_accuracy = np.trace(val_confusion_matrices_absolute[1]) * 100 / np.sum(
                val_confusion_matrices_absolute[1])
            final_val_object_accuracy = np.trace(val_confusion_matrices_absolute[2]) * 100 / np.sum(
                val_confusion_matrices_absolute[2])

        wandb.log({f"Final_training_sentence_wise_accuracy": final_train_sentence_wise_accuracy,
                   f"Final_training_action_accuracy": final_train_action_accuracy,
                   f"Final_training_color_accuracy": final_train_color_accuracy,
                   f"Final_training_object_accuracy": final_train_object_accuracy,

                   f"Final_training_action_actual_predicted_ranks": final_train_rank_percentages[0],
                   f"Final_training_color_actual_predicted_ranks": final_train_rank_percentages[1],
                   f"Final_training_object_actual_predicted_ranks": final_train_rank_percentages[2],

                   f"Final_validation_sentence_wise_accuracy": final_val_sentence_wise_accuracy,
                   f"Final_validation_action_accuracy": final_val_action_accuracy,
                   f"Final_validation_color_accuracy": final_val_color_accuracy,
                   f"Final_validation_object_accuracy": final_val_object_accuracy,

                   f"Final_validation_action_actual_predicted_ranks": final_val_rank_percentages[0],
                   f"Final_validation_color_actual_predicted_ranks": final_val_rank_percentages[1],
                   f"Final_validation_object_actual_predicted_ranks": final_val_rank_percentages[2],

                   f"Final_training_wrong_predictions": final_train_wrong_predictions,
                   f"Final_validation_wrong_predictions": final_val_wrong_predictions})

        cf_matrices_absolute_action = np.zeros((6, 4, 4))
        cf_matrices_absolute_color = np.zeros((6, 6, 6))
        cf_matrices_absolute_object = np.zeros((6, 9, 9))

        cf_matrices_absolute_action_gen = np.zeros((6, 4, 4))
        cf_matrices_absolute_color_gen = np.zeros((6, 6, 6))
        cf_matrices_absolute_object_gen = np.zeros((6, 9, 9))

        for i in range(1, 7):
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

            # dataloader
            test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=False,
                                     num_workers=config["num_workers"], pin_memory=True)
            gen_test_loader = DataLoader(dataset=gen_test_data, batch_size=config["batch_size"], shuffle=False,
                                         num_workers=config["num_workers"])

            (confusion_matrices_absolute, wrong_predictions, sentence_wise_accuracy, rank_percentages, correct_all_timestaps,
            correlation_matrix)= get_evaluation(model,
                                                test_loader,
                                                config["device"],
                                                f"V{i} test")

            (confusion_matrices_absolute_gen, wrong_predictions_gen, sentence_wise_accuracy_gen, rank_percentages_gen,
             correct_all_timestaps_gen, gen_test_correlation_matrix) = get_evaluation(model,
                                                                                     gen_test_loader,
                                                                                     config["device"],
                                                                                     f"V{i} generalization test")


            confusion_matrices_relative = get_relative_confusion_matrix(confusion_matrices_absolute)
            confusion_matrices_relative_gen = get_relative_confusion_matrix(confusion_matrices_absolute_gen)

            plt = create_line_plt(correct_all_timestaps, f"V{i}-test-{data_conf}")
            wandb.log({f"V{i}-test-over_time": wandb.Image(plt)})

            plt = create_line_plt(correct_all_timestaps_gen, f"V{i}-generalization-test-{data_conf}")
            wandb.log({f"V{i}-generalization-test-over_time": wandb.Image(plt)})

            plt = visualize_correlation_matrix(correlation_matrix, f"V{i} test {data_conf}Correlation matrix")
            wandb.log({f"V{i}-test-Correlation_matrix": wandb.Image(plt)})

            plt = visualize_correlation_matrix(gen_test_correlation_matrix, f"V{i} generalization test {data_conf}Correlation matrix")
            wandb.log({f"V{i}-generalization-test-Correlation_matrix": wandb.Image(plt)})



            for n, sentence_part in enumerate(['action', 'color', 'object']):


                plt = create_log_bar_chart_plt(rank_percentages[n], f"V{i}-test-{sentence_part}")
                wandb.log({f"V{i}-test-actual-predicted-{sentence_part}": wandb.Image(plt)})

                plt = create_log_bar_chart_plt(rank_percentages_gen[n], f"V{i}-generalization-test-{sentence_part}")
                wandb.log({f"V{i}-generalization-test-actual-predicted-{sentence_part}": wandb.Image(plt)})

                plt = create_confusion_matrix_plt(confusion_matrices_absolute[n],
                                                  f"V{i}-test-absolute-{sentence_part}-{run_name}", "./logs/", False)
                wandb.log({f"V{i}-test-absolute-{sentence_part}": plt})

                plt = create_confusion_matrix_plt(confusion_matrices_relative[n],
                                                  f"V{i}-test-relative-{sentence_part}-{run_name}", "./logs/", True)
                wandb.log({f"V{i}-test-relative-{sentence_part}": plt})

                plt = create_confusion_matrix_plt(confusion_matrices_absolute_gen[n],
                                                  f"V{i}-generalization-test-absolute-{sentence_part}-{run_name}", "./logs/", False)
                wandb.log({f"V{i}-generalization-test-absolute-{sentence_part}": plt})

                plt = create_confusion_matrix_plt(confusion_matrices_relative_gen[n],
                                                  f"V{i}-generalization-test-relative-{sentence_part}-{run_name}", "./logs/", True)
                wandb.log({f"V{i}-generalization-test-relative-{sentence_part}": plt})

                test_action_accuracy = np.trace(confusion_matrices_absolute[0]) * 100 / np.sum(
                    confusion_matrices_absolute[0])
                test_color_accuracy = np.trace(confusion_matrices_absolute[1]) * 100 / np.sum(
                    confusion_matrices_absolute[1])
                test_object_accuracy = np.trace(confusion_matrices_absolute[2]) * 100 / np.sum(
                    confusion_matrices_absolute[2])

                gen_test_action_accuracy = np.trace(confusion_matrices_absolute_gen[0]) * 100 / np.sum(
                    confusion_matrices_absolute_gen[0])
                gen_test_color_accuracy = np.trace(confusion_matrices_absolute_gen[1]) * 100 / np.sum(
                    confusion_matrices_absolute_gen[1])
                gen_test_object_accuracy = np.trace(confusion_matrices_absolute_gen[2]) * 100 / np.sum(
                    confusion_matrices_absolute_gen[2])

                cf_matrices_absolute_action[i - 1] = confusion_matrices_absolute[0]
                cf_matrices_absolute_color[i - 1] = confusion_matrices_absolute[1]
                cf_matrices_absolute_object[i - 1] = confusion_matrices_absolute[2]
                cf_matrices_absolute_action_gen[i - 1] = confusion_matrices_absolute_gen[0]
                cf_matrices_absolute_color_gen[i - 1] = confusion_matrices_absolute_gen[1]
                cf_matrices_absolute_object_gen[i - 1] = confusion_matrices_absolute_gen[2]

            wandb.log({f"V{i}-test_sentence_wise_accuracy": sentence_wise_accuracy,
                       f"V{i}-test_action_accuracy": test_action_accuracy,
                       f"V{i}-test_color_accuracy": test_color_accuracy,
                       f"V{i}-test_object_accuracy": test_object_accuracy,

                       f"V{i}-test_action_actual_predicted_ranks": rank_percentages[0],
                       f"V{i}-test_color_actual_predicted_ranks": rank_percentages[1],
                       f"V{i}-test_object_actual_predicted_ranks": rank_percentages[2],


                       f"V{i}-generalization_test_sentence_wise_accuracy": sentence_wise_accuracy_gen,
                       f"V{i}-generalization_test_action_accuracy": gen_test_action_accuracy,
                       f"V{i}-generalization_test_color_accuracy": gen_test_color_accuracy,
                       f"V{i}-generalization_test_object_accuracy": gen_test_object_accuracy,

                       f"V{i}-generalizationtest_action_actual_predicted_ranks": rank_percentages_gen[0],
                       f"V{i}-generalizationtest_test_color_actual_predicted_ranks": rank_percentages_gen[1],
                       f"V{i}-generalizationtest_test_object_actual_predicted_ranks": rank_percentages_gen[2],


                       f"V{i}-test_wrong_predictions": wrong_predictions,
                       f"V{i}-generalization_test_wrong_predictions": wrong_predictions_gen})

        test_action_accuracy = np.sum(np.trace(cf_matrices_absolute_action, axis1=1, axis2=2)) * 100 / np.sum(
            cf_matrices_absolute_action)
        test_color_accuracy = np.sum(np.trace(cf_matrices_absolute_color, axis1=1, axis2=2)) * 100 / np.sum(
            cf_matrices_absolute_color)
        test_object_accuracy = np.sum(np.trace(cf_matrices_absolute_object, axis1=1, axis2=2)) * 100 / np.sum(
            cf_matrices_absolute_object)

        gen_test_action_accuracy = np.sum(
            np.trace(cf_matrices_absolute_action_gen, axis1=1, axis2=2)) * 100 / np.sum(
            cf_matrices_absolute_action_gen)
        gen_test_color_accuracy = np.sum(
            np.trace(cf_matrices_absolute_color_gen, axis1=1, axis2=2)) * 100 / np.sum(
            cf_matrices_absolute_color_gen)
        gen_test_object_accuracy = np.sum(
            np.trace(cf_matrices_absolute_object_gen, axis1=1, axis2=2)) * 100 / np.sum(
            cf_matrices_absolute_object_gen)

        wandb.log({"test_action_accuracy": test_action_accuracy,
                   "test_color_accuracy": test_color_accuracy,
                   "test_object_accuracy": test_object_accuracy})

        wandb.log({"generalization_test_action_accuracy": gen_test_action_accuracy,
                   "generalization_test_color_accuracy": gen_test_color_accuracy,
                   "generalization_test_object_accuracy": gen_test_object_accuracy})

        # save
        print_with_time(f"Saving model to {config['save_file']}")
        torch.save(model.state_dict(), config['save_file'])
