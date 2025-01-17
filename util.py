"""
Created on Fri Mar 17 11:07:25 2023
@author: Saeid
"""

import torch
import os
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import softmax
import scipy
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset


def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj


class Dataseting(Dataset):
    """
    A custom dataset class for handling data and target arrays in PyTorch.

    This class inherits from `torch.utils.data.Dataset` and is designed to
    handle datasets where the data and target are numpy arrays. It converts
    these arrays into PyTorch tensors and optionally applies transformations
    to the data.
    """

    def __init__(self, data, target, transform=None):
        """

        Args:
            data (torch.Tensor): The input data converted to a PyTorch tensor.
            target (torch.Tensor): The target labels converted to a PyTorch tensor.
            transform (callable or None): The transformation function applied to the data.
        """
        self.data = torch.from_numpy(data).float()  # convert array into tensor
        self.target = torch.from_numpy(target).long()  # convert array into tensor
        self.transform = transform  # applies transformers

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]

        if self.transform:
            data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.data)


def creating_dataloaders(
    DataDirectory, Training_scene, Test_scene, train_percentage, img_size, batch_size
):
    """
    Creates PyTorch DataLoader objects for training, validation, and testing datasets.

    This function loads training, validation, and testing patches and their corresponding
    labels from specified directories. It splits the training data into training and
    validation sets based on the given train_percentage and formats the data for PyTorch models.

    Args:
        DataDirectory (str): Path to the main data directory containing scene folders.
        Training_scene (str): Name of the folder containing training data (patches and labels).
        Test_scene Name of the folder containing test data (patches and labels).
        train_percentage (float): Percentage of training data to be used for training
        img_size (int): Size of the images (height and width).
        batch_size (int): Batch size to be used for DataLoader objects.

    Returns:
        tuple: A tuple containing:
            - Train_Loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - Val_Loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            - Test_Loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.

    """
    # -- Read Training pathes
    Training_Patches = np.load(
        os.path.join(DataDirectory, Training_scene, "Training_Patches.npy")
    )
    # from  [B, H, W, C]  to  [B, C, H, W]
    Training_Patches = np.swapaxes(Training_Patches, 2, 3)
    Training_Patches = np.swapaxes(Training_Patches, 1, 2)

    # -- Read Training Labels
    Training_Labels = np.load(
        os.path.join(DataDirectory, Training_scene, "Training_Labels.npy")
    )

    # -- Use train_percentage of training sampels to train our model
    Part_Train_Patches, Part_Val_Patches, Part_Train_Labels, part_Val_Labels = (
        train_test_split(
            Training_Patches,
            Training_Labels,
            random_state=42,
            shuffle=True,
            train_size=train_percentage / 100.0,
        )
    )
    print(f"for {train_percentage}% training sampels\n")
    print(Counter(Part_Train_Labels).most_common())
    # -- Creat Training Dtatset
    traindata = Dataseting(Part_Train_Patches, Part_Train_Labels)

    Train_Loader = torch.utils.data.DataLoader(
        traindata, batch_size=batch_size, pin_memory=True, shuffle=True  # noqa
    )

    # -- Creat Validation Dataset
    valdata = Dataseting(Part_Val_Patches, part_Val_Labels)

    Val_Loader = torch.utils.data.DataLoader(
        valdata, batch_size=batch_size, pin_memory=True, shuffle=False  # noqa
    )

    # -- Read test patches
    Test_Patches = np.load(os.path.join(DataDirectory, Test_scene, "Test_Patches.npy"))

    # from  [B, H, W, C]  to  [B, C, H, W]
    Test_Patches = np.swapaxes(Test_Patches, 2, 3)
    Test_Patches = np.swapaxes(Test_Patches, 1, 2)

    # -- Read Test Labels
    Test_Labels = np.load(os.path.join(DataDirectory, Test_scene, "Test_Labels.npy"))

    # -- Creat Test Dataset
    testdata = Dataseting(Test_Patches, Test_Labels)
    Test_Loader = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, pin_memory=True, shuffle=False  # noqa
    )
    return Train_Loader, Val_Loader, Test_Loader


def confusionmatrix(test_loader, model, result_path, patch_size):
    """
    Computes the confusion matrix and various classification metrics for a trained model
    using the given test dataset and saves the results to an Excel file.

    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): PyTorch model to evaluate.
        result_path (str): Path to the directory where the trained model and results are stored.
        patch_size (int): Patch size used during training and evaluation.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- load a trained model
    trained_model_path = os.path.join(
        result_path, ("BestResult_test" + str(patch_size) + "_patchsize.pt")
    )

    model_state = torch.load(trained_model_path)["model_state_dict"]

    # -- upload the trained model's weights
    model.load_state_dict(model_state)
    model = model.to(device)
    # Set the model to "evaluation" mode, b/c we don't want to make any updates!
    model = model.eval()

    # -- compute the output of the trained model for test samples
    with torch.no_grad():
        y_true = []
        y_pred = []  # train each model seperately for one epoch
        for inputs, labels in test_loader:
            # -- move the batch to the device we are using.
            inputs = moveTo(inputs, device)
            labels = moveTo(labels, device)
            y_hat = model(inputs)  # this just computed f_Θ(x(i))
            if isinstance(labels, torch.Tensor):
                # -- moving labels & predictions back to CPU for computing / storing predictions
                labels = labels.detach().cpu().numpy()
                y_hat = y_hat.detach().cpu().numpy()
                # -- add to predictions so far
                y_true.extend(labels.tolist())
                y_pred.extend(y_hat.tolist())

    # -- We have a classification problem, convert to labels
    y_pred = np.asarray(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.asarray(y_true)

    # -- Calculate confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # -- Show the confusion matrix
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=["Forest", "Water", "Dense", "Urban", "Farm"],
    )
    cm_display.plot()
    plt.show()
    # Of the positives predicted, what percentage is truly positive?
    Precision = np.round(metrics.precision_score(y_true, y_pred, average=None), 3)
    # Of all the positive cases, what percentage are predicted positive?
    Sensitivity_recall = np.round(metrics.recall_score(y_true, y_pred, average=None), 3)
    # F-score is the "harmonic mean" of precision and sensitivity.
    F1_Score = np.round(metrics.f1_score(y_true, y_pred, average=None), 3)
    # Accuracy measures how often the model is correct.
    Accuracy = np.round(metrics.accuracy_score(y_true, y_pred), 4)
    # How well the model is at prediciting negative results?
    Specificity = np.round(metrics.recall_score(y_true, y_pred, average=None), 3)

    # -- Save the metrics in a text file
    Class_Names = ["Forest", "Water", "Dense urban", "urban", "Farm"]
    confusion_df = pd.DataFrame(
        confusion_matrix, columns=Class_Names, index=Class_Names
    )
    # -- Add other metrics
    confusion_df.loc["Precision"] = Precision
    confusion_df.loc["Recall"] = Sensitivity_recall
    confusion_df.loc["F1_Score"] = F1_Score
    confusion_df.loc["Specificity"] = Specificity
    confusion_df.loc["Accuracy"] = [Accuracy, "", "", "", ""]

    # -- Save the metrics
    confusion_df.to_excel(os.path.join(result_path, ("confusion_matrix.xlsx")))


def SceneClassification(
    test_scene,
    model,
    result_path,
    patch_size,
    num_classes=6,
    Row_Step=None,
    patchratio=None,
    LandMask=None,
):
    """
    Classifies a given test scene using a trained model and saves the results (classified scene,
    probabilities, and logits) in MAT files.

    Args:
        test_scene (np.ndarray): 4D array test scene to be classified.                          
        model (torch.nn.Module): PyTorch model used for classification.
        result_path (str): Path to the directory where the classification results will be saved.
        patch_size (int): Size of the patches to be extracted from the test scene for classification.
        num_classes (int, optional): Number of classes for classification. Defaults to 6.
        Row_Step (int, optional): Step size for processing rows at a time, used to manage memory.
                                  Defaults to None.
        patchratio (int, optional): Number of patches to process at a time when there is a memory issue.
                                    Defaults to None.
        LandMask (np.ndarray, optional): Mask indicating land areas for classification. Defaults to None.

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    H, W, C, _, _ = test_scene.shape
    # -- Load the trained models
    # -- load a trained model
    trained_model_path = os.path.join(
        result_path, ("BestResult" + str(patch_size) + "_patchsize.pt")
    )

    model_state = torch.load(trained_model_path)["model_state_dict"]
    print("trained weights have been successfully uploaded")
    # -- upload the trained model's weights
    model.load_state_dict(model_state)
    model = model.to(device)
    # Set the model to "evaluation" mode,
    model = model.eval()

    classifeid_scene = np.zeros((H, W)) - 1
    logits = np.zeros((num_classes, H, W))
    probability = np.zeros((num_classes, H, W))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Start_Time = time.time()

    print("Classifying scene starts....")
    # Row_Step=5
    with torch.no_grad():

        # -- When there is no memmory problem by classifying more than one row at once
        if Row_Step is not None and patchratio is None:
            for row in range(0, H, Row_Step):
                if row + Row_Step > H:
                    PATCHES = test_scene[row:H, :, :, :]
                    ROWS = H
                else:
                    PATCHES = test_scene[row : row + Row_Step, :, :, :]
                    ROWS = row + Row_Step
                if row % 200 == 0:
                    print(f"row: {row}/{H}")
                PATCHES = PATCHES.type(torch.cuda.FloatTensor)
                PATCHES = PATCHES.to(device)
                PATCHES = torch.reshape(
                    PATCHES, (-1, C, patch_size, patch_size)
                )  # noqa

                y_hat = model(PATCHES)  # digits / predictions
                # moving predictions back to CPU for computing / storing predictions
                y_hat = y_hat.detach().cpu().numpy()
                y_hat = y_hat.transpose()
                y_hat = y_hat.reshape(num_classes, -1, W)
                # Save the logits
                logits[:, row:ROWS, :] = y_hat
                # Calculate the probabilities
                probability[:, row:ROWS, :] = softmax(y_hat, axis=0)  # noqa
                # Find the class with the highest probability as the predicted class
                Pred = np.argmax(probability[:, row:ROWS, :], axis=0)  # noqa
                classifeid_scene[row:ROWS, :] = Pred

            # -- End of classification all rows

        # -- if memmory allocation problem exists
        # In this case all patches in one row are extracted and divided into several groups
        # Then, each group is classified and saved
        if Row_Step is None and patchratio is not None:
            for row in range(0, H):
                PATCHES = test_scene[row : row + 1, :, :, :]
                if row % 1000 == 0:
                    print(f"row: {row}/{H}")
                PATCHES = PATCHES.type(torch.cuda.FloatTensor)
                PATCHES = PATCHES.to(device)
                PATCHES = torch.reshape(
                    PATCHES, (-1, C, patch_size, patch_size)
                )  # noqa

                for i in range(0, PATCHES.shape[0], patchratio):
                    # print(f'i:{i}   W:{W}')
                    # Extract patchratio patches to not face memory issue
                    if i + patchratio >= W:
                        Col_Right = W
                    else:
                        Col_Right = i + patchratio
                    sub_Patches = PATCHES[i:Col_Right, :, :, :]
                    y_hat = model(sub_Patches)  # digits/predictions
                    y_hat = y_hat.detach().cpu().numpy()
                    y_hat = y_hat.transpose()
                    # Save the logits
                    logits[:, row, i:Col_Right] = y_hat
                    # Calculate the probabilities
                    probability[:, row, i:Col_Right] = softmax(y_hat, axis=0)  # noqa
                    # Find the class with the highest probability as the predicted class
                    Pred = np.argmax(probability[:, row, i:Col_Right], axis=0)  # noqa
                    classifeid_scene[row, i:Col_Right] = Pred  # noqa
                # End of calculating a row
            # End of Classification all rowa
        End_Time = time.time()
        Classification_Time = (End_Time - Start_Time) / 3600
        print(
            "***************   End of Scene Classification   ***************\n"
        )  # noqa
        print(
            "***************   Scene classification took : {:.2f} hours".format(
                Classification_Time
            )
        )  # noqa

        # -- Save the results of each model
        scipy.io.savemat(
            (
                result_path
                + "//"
                + ("Classified_Scene_" + str(patch_size) + "_patchsize.mat")
            ),
            mdict={"Classified_Scene": classifeid_scene},
        )  # noqa
        scipy.io.savemat(
            (
                result_path
                + "//"
                + ("Probabilities_" + str(patch_size) + "_patchsize.mat")
            ),
            mdict={"Probability": probability},
        )  # noqa
        scipy.io.savemat(
            (result_path + "//" + ("logits_" + str(patch_size) + "_patchsize.mat")),
            mdict={"Logits": logits},
        )  # noqa
