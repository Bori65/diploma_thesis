

from models import CompositModel
from dataset import MultimodalSimulation
from evaluation import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime

model_paths = {
    "V1-A4-C1-O4-X-J": "saved/V1-A4-C1-O4-X-myresults/esn_pytorch-resnet18-512-256-V1-A4-C1-O4-X-J-2025-04-03-18-28.pt",
    "V1-A4-C1-O4-X": "saved/V1-A4-C1-O4-X-myresults/esn_pytorch-resnet18-512-256-V1-A4-C1-O4-X-2025-03-13-23-14.pt"}

model_paths1 = {
    "V1-A4-C1-O4-X-J": "saved/V1-A4-C1-O4-X-myresults/esn_pytorch-resnet18-512-256-V1-A4-C1-O4-X-J-2025-04-03-18-28.pt",
    "V1-A4-C1-O4-X": "saved/V1-A4-C1-O4-X-myresults/esn_pytorch-resnet18-512-256-V1-A4-C1-O4-X-2025-03-13-23-14.pt",
    "V1-A4-C6-O4-X-J": "saved/V1-A4-C6-O4-X-myresults/esn_pytorch-resnet18-512-256-V1-A4-C6-O4-X-J-2025-04-11-15-12.pt",
    "V1-A4-C6-O4-X": "saved/V1-A4-C6-O4-X-myresults/esn_pytorch-resnet18-512-256-V1-A4-C6-O4-X-2025-04-04-12-17.pt",
    "V1-A4-C1-O9-J": "saved/V1-A4-C1-O9-myresults/esn_pytorch-resnet18-512-256-V1-A4-C1-O9-J-2025-04-11-15-12.pt",
    "V1-A4-C1-O9": "saved/V1-A4-C1-O9-myresults/esn_pytorch-resnet18-512-256-V1-A4-C1-O9-2025-04-04-12-17.pt",
    "V1-A4-C6-O9-J": "saved/V1-A4-C6-O9-myresults/esn_pytorch-resnet18-512-256-V1-A4-C6-O9-J-2025-04-11-15-12.pt",
    "V1-A4-C6-O9": "saved/V1-A4-C6-O9-myresults/esn_pytorch-resnet18-512-256-V1-A4-C6-O9-2025-04-04-12-17.pt",
    "V2-A4-C1-O4-X-J": "saved/V2-A4-C1-O4-X-myresults/esn_pytorch-resnet18-512-256-V2-A4-C1-O4-X-J-2025-03-31-22-47.pt",
    "V2-A4-C1-O4-X": "saved/V2-A4-C1-O4-X-myresults/esn_pytorch-resnet18-512-256-V2-A4-C1-O4-X-2025-04-12-14-59.pt",
    "V2-A4-C6-O4-X-J": "saved/V2-A4-C6-O4-X-myresults/esn_pytorch-resnet18-512-256-V2-A4-C6-O4-X-J-2025-03-13-18-36.pt",
    "V2-A4-C6-O4-X": "saved/V2-A4-C6-O4-X-myresults/esn_pytorch-resnet18-512-256-V2-A4-C6-O4-X-2025-03-13-18-36.pt",
    "V2-A4-C1-O9-J": "saved/V2-A4-C1-O9-myresults/esn_pytorch-resnet18-512-256-V2-A4-C1-O9-J-2025-03-12-04-53.pt",
    "V2-A4-C1-O9": "saved/V2-A4-C1-O9-myresults/esn_pytorch-resnet18-512-256-V2-A4-C1-O9-2025-03-18-19-27.pt",
    "V2-A4-C6-O9-J": "saved/V2-A4-C6-O9-myresults/esn_pytorch-resnet18-512-256-V2-A4-C6-O9-J-2025-03-15-12-48.pt",
    "V2-A4-C6-O9": "saved/V2-A4-C6-O9-myresults/esn_pytorch-resnet18-512-256-V2-A4-C6-O9-2025-03-16-14-41.pt",
    "V6-A4-C1-O4-X-J": "saved/V6-A4-C1-O4-X-myresults/esn_pytorch-resnet18-512-256-V6-A4-C1-O4-X-J-2025-03-16-19-45.pt",
    "V6-A4-C1-O4-X": "saved/V6-A4-C1-O4-X-myresults/esn_pytorch-resnet18-512-256-V6-A4-C1-O4-X-2025-04-12-20-45.pt",
    "V6-A4-C6-O4-X-J": "saved/V6-A4-C6-O4-X-myresults/esn_pytorch-resnet18-512-256-V6-A4-C6-O4-X-J-2025-04-04-05-18.pt",
    "V6-A4-C6-O4-X": "saved/V6-A4-C6-O4-X-myresults/esn_pytorch-resnet18-512-256-V6-A4-C6-O4-X-2025-03-09-03-47.pt",
    "V6-A4-C1-O9-J": "saved/V6-A4-C1-O9-myresults/esn_pytorch-resnet18-512-256-V6-A4-C1-O9-J-2025-03-15-23-14.pt",
    "V6-A4-C1-O9": "saved/V6-A4-C1-O9-myresults/esn_pytorch-resnet18-512-256-V6-A4-C1-O9-2025-03-15-23-14.pt",
    "V6-A4-C6-O9-J": "saved/V6-A4-C6-O9-myresults/esn_pytorch-resnet18-512-256-V6-A4-C6-O9-J-2025-04-13-03-14.pt",
    "V6-A4-C6-O9": "saved/V6-A4-C6-O9-myresults/esn_pytorch-resnet18-512-256-V6-A4-C6-O9-2025-04-13-03-14.pt"}


def extract_config_from_name(name):
    config = {
        "visible_objects": [int(name[1])],
        "different_actions": int(name[4]),
        "different_colors": int(name[7]),
        "different_objects": int(name[10]),
        "exclusive_colors": "X" in name,
        "no_joints": "J" not in name,
    }
    return config

def run_test(model_path, config_name, data_path):
    base_config = {
        "data_path" : data_path,
        "max_frames": 16,
        "frame_stride": 1,
        "same_size": True,
        "num_training_samples": 5000,
        "num_validation_samples": 2500,
        "image_features": 256,
        "convolutional_features": 1024,
        "hidden_dim": 512,
        "dropout1": 0.5,
        "dropout2": 0.0,
        "freeze": False,
        "leaking_rate": 0.02,
        "lambda_reg": 0.01,
        "spectral_radius": 0.98,
        "output_steps": "last",
        "readout_training": "gd",
        "sequence_architecture": "esn_pytorch",
        "vision_architecture": "resnet18",
        "precooked": False,
        "pretrained": True,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 16,
        "num_workers": 8,
    }
    config = {**base_config, **extract_config_from_name(config_name)}

    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
    data_conf = f"V{config['visible_objects'][0]}-A{config['different_actions']}-C{config['different_colors']}-" \
                f"O{config['different_objects']}-{'X-' if config['exclusive_colors'] else ''}" \
                f"{'J-' if not config['no_joints'] else ''}"
    # dataloader
    train_loader = DataLoader(dataset=training_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"])
    val_loader = DataLoader(dataset=validation_data, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"])


    model = CompositModel(
        vision_architecture=config["vision_architecture"],
        pretrained_vision=config["pretrained"],
        dropout1=config["dropout1"],
        dropout2=config["dropout2"],
        image_features=config["image_features"],
        seq2seq_architecture=config["sequence_architecture"],
        hidden_dim=config["hidden_dim"],
        freeze=config["freeze"],
        leaking_rate=config["leaking_rate"],
        lambda_reg=config["lambda_reg"],
        spectral_radius=config["spectral_radius"],
        output_steps=config["output_steps"],
        readout_training=config["readout_training"],
        precooked=config["precooked"],
        convolutional_features=config["convolutional_features"],
        no_joints=config["no_joints"],
    )

    model.load_state_dict(torch.load(model_path))
    model.to(config["device"])

    run_name = f"Test-{config_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project="Comp gen with TorchESN", name=run_name, config=config)

    (train_confusion_matrices_absolute, final_train_wrong_predictions, final_train_sentence_wise_accuracy,
     final_train_rank_percentages, final_train_correct_all_timestaps, final_train_correlation_matrix) = get_evaluation(
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

    plt = create_line_plt(final_val_correct_all_timestaps, f"Final-validation-{data_conf}")
    wandb.log({f"Final-validation-over-time": wandb.Image(plt)})

    plt = visualize_correlation_matrix(final_train_correlation_matrix, f"Final training {data_conf}Correlation matrix")
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

        # final_train_accuracy = np.trace(train_confusion_matrix_absolute) * 100 / np.sum(train_confusion_matrix_absolute)
        final_train_action_accuracy = np.trace(train_confusion_matrices_absolute[0]) * 100 / np.sum(
            train_confusion_matrices_absolute[0])
        final_train_color_accuracy = np.trace(train_confusion_matrices_absolute[1]) * 100 / np.sum(
            train_confusion_matrices_absolute[1])
        final_train_object_accuracy = np.trace(train_confusion_matrices_absolute[2]) * 100 / np.sum(
            train_confusion_matrices_absolute[2])

        # final_val_accuracy = np.trace(val_confusion_matrix_absolute) * 100 / np.sum(val_confusion_matrix_absolute)
        final_val_action_accuracy = np.trace(val_confusion_matrices_absolute[0]) * 100 / np.sum(
            val_confusion_matrices_absolute[0])
        final_val_color_accuracy = np.trace(val_confusion_matrices_absolute[1]) * 100 / np.sum(
            val_confusion_matrices_absolute[1])
        final_val_object_accuracy = np.trace(val_confusion_matrices_absolute[2]) * 100 / np.sum(
            val_confusion_matrices_absolute[2])

    wandb.log({f"Final_training_sentence_wise_accuracy": final_train_sentence_wise_accuracy,
               # f"Final_training_accuracy": final_train_accuracy,
               f"Final_training_action_accuracy": final_train_action_accuracy,
               f"Final_training_color_accuracy": final_train_color_accuracy,
               f"Final_training_object_accuracy": final_train_object_accuracy,

               f"Final_training_action_actual_predicted_ranks": final_train_rank_percentages[0],
               f"Final_training_color_actual_predicted_ranks": final_train_rank_percentages[1],
               f"Final_training_object_actual_predicted_ranks": final_train_rank_percentages[2],

               f"Final_validation_sentence_wise_accuracy": final_val_sentence_wise_accuracy,
               # f"Final_validation_accuracy": final_val_accuracy,
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

    for i in [1,2,6]:
        # for i in [2]:
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

        (
        confusion_matrices_absolute, wrong_predictions, sentence_wise_accuracy, rank_percentages, correct_all_timestaps,
        correlation_matrix) = get_evaluation(model,
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

        plt = visualize_correlation_matrix(gen_test_correlation_matrix,
                                           f"V{i} generalization test {data_conf}Correlation matrix")
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
                                              f"V{i}-generalization-test-absolute-{sentence_part}-{run_name}",
                                              "./logs/", False)
            wandb.log({f"V{i}-generalization-test-absolute-{sentence_part}": plt})

            plt = create_confusion_matrix_plt(confusion_matrices_relative_gen[n],
                                              f"V{i}-generalization-test-relative-{sentence_part}-{run_name}",
                                              "./logs/", True)
            wandb.log({f"V{i}-generalization-test-relative-{sentence_part}": plt})

            # test_accuracy = np.trace(confusion_matrix_absolute) * 100 / np.sum(confusion_matrix_absolute)
            test_action_accuracy = np.trace(confusion_matrices_absolute[0]) * 100 / np.sum(
                confusion_matrices_absolute[0])
            test_color_accuracy = np.trace(confusion_matrices_absolute[1]) * 100 / np.sum(
                confusion_matrices_absolute[1])
            test_object_accuracy = np.trace(confusion_matrices_absolute[2]) * 100 / np.sum(
                confusion_matrices_absolute[2])

            # gen_test_accuracy = np.trace(confusion_matrix_absolute_gen) * 100 / np.sum(confusion_matrix_absolute_gen)
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
                   # f"V{i}-test_accuracy": test_accuracy,
                   f"V{i}-test_action_accuracy": test_action_accuracy,
                   f"V{i}-test_color_accuracy": test_color_accuracy,
                   f"V{i}-test_object_accuracy": test_object_accuracy,

                   f"V{i}-test_action_actual_predicted_ranks": rank_percentages[0],
                   f"V{i}-test_color_actual_predicted_ranks": rank_percentages[1],
                   f"V{i}-test_object_actual_predicted_ranks": rank_percentages[2],

                   f"V{i}-generalization_test_sentence_wise_accuracy": sentence_wise_accuracy_gen,
                   # f"V{i}-generalization_test_accuracy": gen_test_accuracy,
                   f"V{i}-generalization_test_action_accuracy": gen_test_action_accuracy,
                   f"V{i}-generalization_test_color_accuracy": gen_test_color_accuracy,
                   f"V{i}-generalization_test_object_accuracy": gen_test_object_accuracy,

                   f"V{i}-generalizationtest_action_actual_predicted_ranks": rank_percentages_gen[0],
                   f"V{i}-generalizationtest_test_color_actual_predicted_ranks": rank_percentages_gen[1],
                   f"V{i}-generalizationtest_test_object_actual_predicted_ranks": rank_percentages_gen[2],

                   f"V{i}-test_wrong_predictions": wrong_predictions,
                   f"V{i}-generalization_test_wrong_predictions": wrong_predictions_gen})

    # test_accuracy = np.sum(np.trace(cf_matrices_absolute, axis1=1, axis2=2)) * 100 / np.sum(cf_matrices_absolute)
    test_action_accuracy = np.sum(np.trace(cf_matrices_absolute_action, axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_action)
    test_color_accuracy = np.sum(np.trace(cf_matrices_absolute_color, axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_color)
    test_object_accuracy = np.sum(np.trace(cf_matrices_absolute_object, axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_object)

    # gen_test_accuracy = np.sum(np.trace(cf_matrices_absolute_gen, axis1=1, axis2=2)) * 100 / np.sum(
    # cf_matrices_absolute_gen)
    gen_test_action_accuracy = np.sum(
        np.trace(cf_matrices_absolute_action_gen, axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_action_gen)
    gen_test_color_accuracy = np.sum(
        np.trace(cf_matrices_absolute_color_gen, axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_color_gen)
    gen_test_object_accuracy = np.sum(
        np.trace(cf_matrices_absolute_object_gen, axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_object_gen)

    wandb.log({  # "test_accuracy": test_accuracy,
        "test_action_accuracy": test_action_accuracy,
        "test_color_accuracy": test_color_accuracy,
        "test_object_accuracy": test_object_accuracy})
    # print_with_time(f"Test accuracy: {test_accuracy:8.4f}%")


    wandb.log({  # "generalization_test_accuracy": gen_test_accuracy,
        "generalization_test_action_accuracy": gen_test_action_accuracy,
        "generalization_test_color_accuracy": gen_test_color_accuracy,
        "generalization_test_object_accuracy": gen_test_object_accuracy})
    # print_with_time(f"Generalization test accuracy: {gen_test_accuracy:8.4f}%")

if __name__ == "__main__":
    DATA_PATH = "../../final-dataset/final-dataset/"

    for config_name, model_path in model_paths.items():
        print(f"Running test for {config_name}...")
        run_test(model_path, config_name, DATA_PATH)

