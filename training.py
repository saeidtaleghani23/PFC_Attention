
import torch
import time
import numpy as np
import os
from util import moveTo


def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    """
    model -- the PyTorch model / "Module" to run for one epoch
    optimizer -- the object that will update the weights of the network
    data_loader -- DataLoader object that returns tuples of (input, label) pairs. 
    loss_func -- the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    device -- the compute lodation to perform training
    score_funcs -- a dictionary of scoring functions to use to evalue the performance of the model
    prefix -- a string to pre-fix to any scores placed into the _results_ dictionary. 
    desc -- a description to use for the progress bar.     
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

        if prefix == "validation" or prefix == "test":
            inputs.requires_grad_(False)  # Ensure inputs don't track gradients
            labels.requires_grad_(False)  # Ensure labels don't track gradients

        # -- Output of the model
        y_hat = model(inputs)

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
            # -- moving labels & predictions back to CPU for computing / storing predictions
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
    return end-start  # time spent on epoch


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- Create Result file
    if os.path.exists(result_path) is not True:
        os.mkdir(result_path)

    # -- save all results
    checkpoint_file_results = os.path.join(
        result_path, ('All_results_'+str(patch_size) + '_patchsize.pt'))
    # -- save the best result based on validation accuracy
    checkpoint_file_best_result = os.path.join(
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
                                      score_funcs, prefix="train", desc="Training")

        # -- Save epoch and processing time
        results["epoch"].append(epoch)
        results["total time"].append(total_train_time)

        #   ******  Validating  ******
        if validation_loader is not None:
            model = model.eval()  # Set the model to "evaluation" mode
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
        # show the progress and metrics
        print('\nEpoch: {}   Training accuracy: {:.2f}   Validation accuracy: {:.2f}   Test Accuracy: {:.2f}'
              .format(epoch, results['train Accuracy'][-1]*100, results['validation Accuracy'][-1]*100, results['test Accuracy'][-1]*100))
        # save the model based on the validation accuracy
        if results['validation Accuracy'][-1] > Best_validation_Accuracy:
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
