"""
Created on Fri Mar 17 11:07:25 2023
@author: Saeid

 num_workers = train_options['num_workers'], 
                                        pin_memory = True)

"""


# %% Libraries

import torch
import os
import time
#import tqdm
import numpy as np
from os.path import join
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import softmax
import scipy.io
from sklearn.model_selection import train_test_split
import Dataset
from collections import Counter

# %% Function


def moveTo(obj, device):
    """
   Moves a Python object or its contents to the specified device.

    Parameters
    ----------
    obj : any
        The Python object to move to a device, or to move its contents to a device.
        This can be a tensor, list, tuple, set, or dictionary. If it is a container
        (list, tuple, set, dictionary), its elements will be recursively moved to the device.
    device : torch.device
        The compute device to move objects to (e.g., 'cpu' or 'cuda').

    Returns
    -------
    any
        The object or container with its contents moved to the specified device. The returned
        type will match the input type.
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


def run_epoch(model,
              optimizer,
              data_loader,
              loss_func,
              device,
              results,
              score_funcs,
              prefix="",
              desc=None):
    """
    This function runs one epoch to train a model.

    Parameters
    ----------
    model : nn.Module
        PFC model that utilizes fine- and course-attention to classify different
        objects in RADARSAT Comact Polarimetric SAR Data.
    optimizer : torch. optim.Optimizer
        Optimizer used for updating model parameters.
    data_loader : DataLoader
        DataLoader providing the data for the current epoch and 
        returns tuples of (input, label) pairs.
    loss_func : callable
        The loss function that takes in two arguments, 
        the model outputs and the labels, and returns a score
    device : torch.device
        Device on which the model and data are located (e.g., 'cpu' or 'cuda').
    results : dict
        Dictionary to store results, such as loss and metrics.
    score_funcs : dict
        Dictionary of scoring functions used to evaluate model performance.
    prefix : str, optional
        Prefix for logging and output (default is "").
    desc : str, optional
        Description for progress bar (default is None).

    Returns
    -------
    process time: float
        time spent on one epoch.

    """
    model = model.to(device)
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in (data_loader):
        # -- Move the batch to the device we are using.
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        # -- Output of the model
        y_hat = model(inputs)  # this just computed f_Θ(x(i))

        # -- Compute loss.
        loss = loss_func(y_hat, labels)

        # -- Training?
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # -- Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            # -- moving labels & predictions back to CPU
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            # -- add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    # -- end of one training epoch
    end = time.time()

    y_pred = np.asarray(y_pred)
    # We have a classification problem, convert to labels
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    # Else, we assume we are working on a regression problem

    results[prefix + " loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append(score_func(y_true, y_pred))
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end-start


def train_model(epoches,
                model,
                optimizer,
                train_loader,
                loss_func,
                score_funcs,
                result_path,
                patch_size,
                validation_loader=None,
                test_loader=None):
    """
    Trains the model for a given number of epochs, with optional validation 
    and testing, and saves the results.

    Parameters
    ----------
    epoches : int
        Number of epochs to train the model.
    model : nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating the model parameters.
    train_loader : DataLoader
        DataLoader providing the training data.
    loss_func : callable
        The loss function used to calculate the training loss.
    score_funcs : dict
        Dictionary of scoring functions used to evaluate model performance.
    result_path : str
        Path to the directory where the results and model checkpoints will be saved.
    patch_size : int
        The size of the patches used in the model.
    validation_loader : DataLoader, optional
        DataLoader providing the validation data, if any (default is None).
    test_loader : DataLoader, optional
        DataLoader providing the test data, if any (default is None).

    Returns
    -------
    None
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- Create Result file
    if os.path.exists(result_path) is not True:
        os.mkdir(result_path)

    # -- save all results
    checkpoint_file_results = join(
        result_path, ('AllResults_' + str(patch_size) + '_patchsize.pt'))
    # -- save the best result based on validation accuracy
    checkpoint_file_best_result = join(
        result_path, ('BestResult_' + str(patch_size) + '_patchsize.pt'))

    # -- send model on the device
    model = model.to(device)
    to_track = ["epoch", "total time", "train Accuracy", "train loss"]

    # -- There is Validation loader?
    if validation_loader is not None:
        to_track.append("validation Accuracy")
        to_track.append("validation loss")

    # -- There is test loader ?
    if test_loader is not None:
        to_track.append("test Accuracy")
        to_track.append("test loss")

    total_train_time = 0
    results = {}

    # -- Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    Best_validation_Accuracy = 0.0

    # -- Train model
    print('Training begins...\n')

    for epoch in range(epoches):
        # -- set the model on train
        model = model.train()
        # -- Train for one epoch
        total_train_time += run_epoch(model, optimizer, train_loader,
                                      loss_func, device, results,
                                      score_funcs, prefix="train",
                                      desc="Training")

        # -- Save epoch and processing time
        results["epoch"].append(epoch)
        results["total time"].append(total_train_time)

        #   ******  Validating  ******
        if validation_loader is not None:
            # Set the model to "evaluation" mode
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, validation_loader,
                          loss_func, device, results,
                          score_funcs, prefix="validation", desc="Validating")

        #   ******  Testing  ******
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader,
                          loss_func, device, results,
                          score_funcs, prefix="test", desc="Testing")

        #   ******  Save results of each epoch  ******
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results
        }, checkpoint_file_results)

        if results['validation Accuracy'][-1] > Best_validation_Accuracy or results['validation Accuracy'][-1] == 1:
            print('\nEpoch: {}   Training accuracy: {:.2f}   best Val accuracy: {:.2f}   Test Accuracy: {:.2f}'
                  .format(epoch, results['train Accuracy'][-1]*100, results['validation Accuracy'][-1]*100, results['test Accuracy'][-1]*100))
            Best_validation_Accuracy = results['validation Accuracy'][-1]
            best_result = {}
            best_result["epoch"] = []
            best_result["train accuracy"] = []
            best_result["validation accuracy"] = []
            best_result["test accuracy"] = []

            best_result["epoch"].append(epoch)
            best_result["train accuracy"].append(results['train Accuracy'][-1])
            best_result["validation accuracy"].append(
                results['validation Accuracy'][-1])
            best_result["test accuracy"].append(results['test Accuracy'][-1])
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': best_result
            }, checkpoint_file_best_result)


def creating_dataloaders(DataDirectory,
                         Training_scene,
                         Test_scene,
                         train_percentage,
                         img_size,
                         batch_size):
    """
    Creates and returns DataLoaders for training, validation, and testing datasets.

    Parameters
    ----------
    DataDirectory : str
        The root directory where the data is stored.
    Training_scene : str
        The name of the training scene directory.
    Test_scene : str
        The name of the test scene directory.
    train_percentage : float
        The percentage of training samples to use for training the model. The remaining samples will be used for validation.
    img_size : int
        The size of the image patches (assumed to be square).
    batch_size : int
        The batch size for the DataLoaders.

    Returns
    -------
    Train_Loader : DataLoader
        DataLoader for the training dataset.
    Val_Loader : DataLoader
        DataLoader for the validation dataset.
    Test_Loader : DataLoader
        DataLoader for the test dataset.
    """

    # -- Read Training pathes
    Training_Patches = np.load(os.path.join(DataDirectory, Training_scene,
                                            'Features',
                                            'Amplitude',
                                            'Standerized dB',
                                            'Training',
                                            (str(img_size) + 'x' + str(img_size)),
                                            'Training_Patches.npy'))
    # from  [B, H, W, C]  to  [B, C, H, W]
    Training_Patches = np.swapaxes(Training_Patches, 2, 3)
    Training_Patches = np.swapaxes(Training_Patches, 1, 2)

    # -- Read Training Labels
    Training_Labels = np.load(os.path.join(DataDirectory,
                                           Training_scene,
                                           'Features',
                                           'Amplitude',
                                           'Standerized dB',
                                           'Training',
                                           (str(img_size) + 'x' + str(img_size)),
                                           'Training_Labels.npy'))

    # -- Use train_percentage of training sampels to train our model
    Part_Train_Patches, Part_Val_Patches, Part_Train_Labels, part_Val_Labels = train_test_split(Training_Patches, Training_Labels,
                                                                                                random_state=42,
                                                                                                shuffle=True,
                                                                                                train_size=train_percentage/100.0)
    print(f'for {train_percentage}% training sampels\n')
    print(Counter(Part_Train_Labels).most_common())
    # -- Creat Training Dtatset
    traindata = Dataset.Dataseting(Part_Train_Patches, Part_Train_Labels)

    Train_Loader = torch.utils.data.DataLoader(traindata,
                                               batch_size=batch_size,  # noqa
                                               pin_memory=True,
                                               shuffle=True  # To keep same samples for all models
                                               )

    # -- Creat Validation Dataset
    valdata = Dataset.Dataseting(Part_Val_Patches,  part_Val_Labels)

    Val_Loader = torch.utils.data.DataLoader(valdata,
                                              batch_size=batch_size,  # noqa
                                              pin_memory=True,
                                              shuffle=True  # To keep same samples for all models
                                              )

    # -- Read test patches
    Test_Patches = np.load(os.path.join(DataDirectory,
                                        Test_scene,
                                        'Features',
                                        'Amplitude',
                                        'Standerized dB',
                                        'Testing',
                                        (str(img_size) + 'x' + str(img_size)),
                                        'Test_Patches.npy'))

    # from  [B, H, W, C]  to  [B, C, H, W]
    Test_Patches = np.swapaxes(Test_Patches, 2, 3)
    Test_Patches = np.swapaxes(Test_Patches, 1, 2)

    # -- Read Test Labels
    Test_Labels = np.load(os.path.join(DataDirectory,
                                       Test_scene,
                                       'Features',
                                       'Amplitude',
                                       'Standerized dB',
                                       'Testing',
                                       (str(img_size) + 'x' + str(img_size)),
                                       'Test_Labels.npy'))

    # -- Creat Test Dataset
    testdata = Dataset.Dataseting(Test_Patches, Test_Labels)
    Test_Loader = torch.utils.data.DataLoader(testdata,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False
                                              )
    return Train_Loader, Val_Loader, Test_Loader


def confusionmatrix(test_loader, model, result_path, patch_size):
    """
    Computes and displays the confusion matrix for a trained model on a test dataset.

    Parameters
    ----------
    test_loader : DataLoader
        DataLoader for the test dataset.
    model : nn.Module
        Trained model to evaluate.
    result_path : str
        Path to the directory where the trained model and results are saved.
    patch_size : int
        The size of the image patches (assumed to be square).

    Returns
    -------
    None
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- load a trained model
    trained_model_path = join(
        result_path, ('BestResult_test' + str(patch_size) + '_patchsize.pt'))

    model_state = torch.load(trained_model_path)['model_state_dict']

    # -- upload the trained model's weights
    model.load_state_dict(model_state)
    model = model.to(device)
    # Set the model to "evaluation" mode
    model = model.eval()

    with torch.no_grad():
        y_true = []
        y_pred = []
        for inputs, labels in (test_loader):
            # -- move the batch to the device we are using.
            inputs = moveTo(inputs, device)
            labels = moveTo(labels, device)
            y_hat = model(inputs)
            if isinstance(labels, torch.Tensor):
                # -- moving labels & predictions back to CPU
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
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=['Forest', 'Water', 'Dense', 'Urban', 'Farm'])
    cm_display.plot()
    plt.show()
    # Of the positives predicted, what percentage is truly positive?
    Precision = np.round(metrics.precision_score(
        y_true, y_pred, average=None), 3)
    # Of all the positive cases, what percentage are predicted positive?
    Sensitivity_recall = np.round(
        metrics.recall_score(y_true, y_pred, average=None), 3)
    # F-score is the "harmonic mean" of precision and sensitivity.
    F1_Score = np.round(metrics.f1_score(y_true, y_pred, average=None), 3)
    # Accuracy measures how often the model is correct.
    Accuracy = np.round(metrics.accuracy_score(y_true, y_pred), 4)
    # How well the model is at prediciting negative results?
    Specificity = np.round(metrics.recall_score(
        y_true, y_pred, average=None), 3)

    # -- Save the metrics in a text file
    Class_Names = ['Forest', 'Water', 'Dense urban', 'urban', 'Farm']
    confusion_df = pd.DataFrame(
        confusion_matrix, columns=Class_Names, index=Class_Names)
    # -- Add other metrics
    confusion_df.loc['Precision'] = Precision
    confusion_df.loc['Recall'] = Sensitivity_recall
    confusion_df.loc['F1_Score'] = F1_Score
    confusion_df.loc['Specificity'] = Specificity
    confusion_df.loc['Accuracy'] = [Accuracy, '', '', '', '']

    # -- Save the metrics
    confusion_df.to_excel(join(result_path, ('confusion_matrix.xlsx')))
    return


def SceneClassification(test_scene,
                        model,
                        result_path,
                        patch_size,
                        num_classes=6,
                        Row_Step=6,
                        patchratio=None,
                        ):
    """
    Classifies the input scene using a trained model and saves the results.

    Parameters
    ----------
    test_scene : torch.Tensor
        The input scene to be classified with shape (H, W, C, _, _), 
        where H is the height, W is the width, and C is the number of channels.
    model : nn.Module
        The trained model to use for classification.
    result_path : str
        The path to the directory where the trained model and results will be saved.
    patch_size : int
        The size of the image patches (assumed to be square).
    num_classes : int, optional
        The number of classes for classification (default is 6).
    Row_Step : int, optional
        The number of rows to classify at once (default is 6). If None, `patchratio` is used instead.
    patchratio : int, optional
        The number of patches to classify at once if memory allocation is an issue. Used when `Row_Step` is None.

    Returns
    -------
    None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    H, W, C, _, _ = test_scene.shape
    # -- load a trained model
    trained_model_path = join(
        result_path, ('BestResult' + str(patch_size) + '_patchsize.pt'))

    model_state = torch.load(trained_model_path)['model_state_dict']
    print('trained weights have been successfully uploaded')
    # -- upload the trained model's weights
    model.load_state_dict(model_state)
    model = model.to(device)
    # Set the model to "evaluation" mode
    model = model.eval()

    classifeid_scene = np.zeros((H, W))-1
    logits = np.zeros((num_classes, H, W))
    probability = np.zeros((num_classes, H, W))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Start_Time = time.time()

    print('Classifying scene starts....')
    with torch.no_grad():
        # -- When there is no memmory problem by classifying more than one row at once
        if Row_Step is not None and patchratio is None:
            for row in range(0, H, Row_Step):
                if row+Row_Step > H:
                    PATCHES = test_scene[row:H, :, :, :]
                    ROWS = H
                else:
                    PATCHES = test_scene[row: row+Row_Step, :, :, :]
                    ROWS = row + Row_Step
                if row % 200 == 0:
                    print(f'row: {row}/{H}')
                PATCHES = PATCHES.type(torch.cuda.FloatTensor)
                PATCHES = PATCHES.to(device)
                PATCHES = torch.reshape(PATCHES, (-1, C, patch_size, patch_size))  # noqa

                y_hat = model(PATCHES)  # digits / predictions
                # moving predictions back to CPU
                y_hat = y_hat.detach().cpu().numpy()
                y_hat = y_hat.transpose()
                y_hat = y_hat.reshape(num_classes, -1, W)
                # Save the logits
                logits[:, row:ROWS, :] = y_hat
                # Calculate the probabilities
                probability[:, row:ROWS, :] = softmax(y_hat, axis=0)
                # Find the class with the highest probability
                Pred = np.argmax(probability[:, row:ROWS, :], axis=0)
                classifeid_scene[row:ROWS, :] = Pred

        # -- if memmory allocation problem exists
        # In this case all patches in one row are extracted and divided
        # into several groups. Then, each group is classified and saved
        if Row_Step is None and patchratio is not None:
            for row in range(0, H):
                PATCHES = test_scene[row: row+1, :, :, :]
                if row % 1000 == 0:
                    print(f'row: {row}/{H}')
                PATCHES = PATCHES.type(torch.cuda.FloatTensor)
                PATCHES = PATCHES.to(device)
                PATCHES = torch.reshape(PATCHES, (-1, C, patch_size, patch_size))  # noqa

                for i in range(0, PATCHES.shape[0], patchratio):
                    # Extract patchratio patches to not face memory issue
                    if i+patchratio >= W:
                        Col_Right = W
                    else:
                        Col_Right = i+patchratio
                    sub_Patches = PATCHES[i: Col_Right, :, :, :]
                    y_hat = model(sub_Patches)  # digits/predictions
                    y_hat = y_hat.detach().cpu().numpy()
                    y_hat = y_hat.transpose()
                    # Save the logits
                    logits[:, row, i:Col_Right] = y_hat
                    # Calculate the probabilities
                    probability[:, row, i:Col_Right] = softmax(y_hat, axis=0)
                    # Find the class with the highest probability
                    Pred = np.argmax(probability[:, row, i:Col_Right], axis=0)
                    classifeid_scene[row, i:Col_Right] = Pred

        End_Time = time.time()
        Classification_Time = (End_Time-Start_Time)/3600
        print('End of Scene Classification\n')
        print(f'Scene classification took : {Classification_Time:.2f} hours')

        # -- Save the results of each model
        scipy.io.savemat((result_path + '//' + ('Classified_Scene_' + str(patch_size) +
                         '_patchsize.mat')), mdict={'Classified_Scene': classifeid_scene})
        scipy.io.savemat((result_path + '//' + ('Probabilities_' + str(patch_size) +
                         '_patchsize.mat')), mdict={'Probability': probability})
        scipy.io.savemat((result_path + '//' + ('logits_' +
                         str(patch_size) + '_patchsize.mat')), mdict={'Logits': logits})
    return
