import os
import torch.nn as nn
import time
import torch
from sklearn.metrics import accuracy_score
from generating_patches import Getting_Patches
from model import pfc_classifier
from training import train_model
from torch import optim as optim
from util import confusionmatrix, SceneClassification, creating_dataloaders
import yaml


if __name__ == "__main__":
    # Define the path to your YAML file
    yaml_file_path = "./config/pfc_config.yaml"

    # Open and read the YAML file
    with open(yaml_file_path, "r") as file:
        pfc_config = yaml.load(file, Loader=yaml.FullLoader)

    # Print the loaded data
    print(pfc_config)

    FileDirectory = os.getcwd()
    DataDirectory = os.path.join(FileDirectory, "Data")

    Result_Directory = os.path.join(
        FileDirectory,
        "results of all models",
        "Training on Ottawa and Test on Quebec",
        "PFC",
    )

    if os.path.exists(Result_Directory) is not True:
        os.mkdir(Result_Directory)

    score_funcs = {"Accuracy": accuracy_score}

    # -- Create training, validating, and testing pathces
    Getting_Patches(
        DataDirectory=DataDirectory,
        PatchSize=pfc_config["MODEL"]["img_size"],
        Training_scene=pfc_config["DATA"]["train_scene"],
        Test_scene=pfc_config["DATA"]["test_scene"],
    )

    # -- Creat a path to save the result
    result_path = os.path.join(
        Result_Directory,
        str(pfc_config["DATA"]["train_percentage"]) + "_percentage training sampels",
    )

    # -- make sure the result_path is available
    if os.path.exists(result_path) is not True:
        os.mkdir(result_path)

    # create a pfc mdoel
    model = pfc_classifier(pfc_config)

    # -- define the optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=pfc_config["TRAIN"]["lr"],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.05,
    )
    # -- create loaders
    Train_Loader, Val_Loader, Test_Loader = creating_dataloaders(
        DataDirectory=DataDirectory,
        Training_scene=pfc_config["DATA"]["train_scene"],
        Test_scene=pfc_config["DATA"]["test_scene"],
        train_percentage=pfc_config["DATA"]["train_percentage"],
        img_size=pfc_config["MODEL"]["img_size"],
        batch_size=pfc_config["TRAIN"]["batch_size"],
    )

    Start = time.time()

    # -- Train the model
    train_model(
        pfc_config["TRAIN"]["epochs"],
        model,
        optimizer,
        Train_Loader,
        nn.CrossEntropyLoss(),
        score_funcs,
        result_path,
        pfc_config["MODEL"]["img_size"],
        validation_loader=Val_Loader,
        test_loader=Test_Loader,
    )

    End = time.time()
    Diff_hrs = (End - Start) / 3600
    print("***********      End of Training        **************")
    print("\n It tooke: {:.3f} hours".format(Diff_hrs))
    # -- Measure confusion matrix
    confusionmatrix(Test_Loader, model, result_path, pfc_config["MODEL"]["img_size"])

    # -- Scene Classification
    print("\n Scene Classification starts \n")
    Start = time.time()

    # -- Read test scene patches created using Getting_Patches
    TestScene = torch.load(
        os.path.join(
            DataDirectory, pfc_config["DATA"]["test_scene"], "test_scene_patches.pt"
        )
    )

    SceneClassification(
        TestScene,
        model,
        result_path,
        pfc_config["MODEL"]["img_size"],
        num_classes=pfc_config["MODEL"]["num_classes"],
        Row_Step=None,
        patchratio=100,
        LandMask=None,
    )

    End = time.time()
    Diff_hrs = (End - Start) / 3600
    print("***********      End of Scene Classificaiton        **************")
    print("\n It tooke: {:.3f} hours".format(Diff_hrs))
