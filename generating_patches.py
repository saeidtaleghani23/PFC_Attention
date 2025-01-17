"""
Reading Labeled Samples in each Scene and extract a patch around each one
"""

import scipy.io
import numpy as np

# from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
from scipy import io
import os
import torch


def Extract_Patches(Data: float, GT: int, Patchsize: int) -> [list, list]:
    """
    Extracts patches and corresponding labels around each ground truth sample.

    Args:
        Data (float): Array containing the data.
        GT (int): Ground truth scene with the same size as Data.
        Patchsize (int): Size of the patch to be extracted around a labeled sample.

    Returns:
        list: Array of patches, with each patch centered around a labeled sample.
        list: Array of labels corresponding to the patches.

    """
    # -- Pad size
    padd_size = Patchsize // 2
    if Patchsize % 2 == 0:  # even pad size
        Padding = ((padd_size - 1, padd_size), (padd_size - 1, padd_size), (0, 0))
        Up_Row = padd_size - 1
        Down_Row = padd_size + 1

        Left_Col = padd_size - 1
        Right_Col = padd_size + 1
    else:  # odd size pad size
        Padding = ((padd_size, padd_size), (padd_size, padd_size), (0, 0))
        Up_Row = padd_size
        Down_Row = padd_size + 1

        Left_Col = padd_size
        Right_Col = padd_size + 1

    # -- Padding Scene, Data, and Ground truth
    Data = np.pad(Data, pad_width=Padding, mode="edge")
    GT = np.pad(GT, pad_width=Padding[0:2], mode="constant", constant_values=0)

    # -- Find indices of labeled pixels
    indices = np.transpose(np.nonzero(GT))

    # -- define labels and patch arrays for each labeled pixel
    Labels = np.zeros((indices.shape[0]))
    Patches = np.zeros((indices.shape[0], Patchsize, Patchsize, Data.shape[-1]))

    for i, loc in enumerate(indices):
        # -- extract a patch around a labeled sample
        patch = Data[
            loc[0] - Up_Row : loc[0] + Down_Row,
            loc[1] - Left_Col : loc[1] + Right_Col,
            :,
        ]
        # -- save the patch into Patches array
        Patches[i, :, :, :] = patch
        # -- extract label of the sample
        Labels[i] = GT[loc[0], loc[1]]

    # -- Return patches along with their labels
    return np.float32(Patches), np.int8(Labels)


def view_as_windows_torch(image: float, shape: tuple, stride=None) -> list:
    """
    Views a tensor as overlapping rectangular windows with a specified stride.

    Args:
        image (torch.Tensor): 4D image tensor, where the last two dimensions represent the image dimensions.
        shape (tuple of int): Shape of the window.
        stride (tuple of int, optional): Stride of the windows. Defaults to half of the window size.

    Returns:
        torch.Tensor: Tensor containing overlapping windows.

    """
    if stride is None:
        stride = shape[0] // 2, shape[1] // 2

    windows = image.unfold(2, shape[0], stride[0])
    return windows.unfold(3, shape[1], stride[1])


def saving_patches(
    Data: list, GT: list, Patchsize: int, root: str, Purpose="training"
) -> None:
    """
    Saves patches extracted around each ground truth sample.

    Args:
        Data (list): ndarray containing the data, which can include a covariance matrix or other features.
        GT (list): Ground truth scene corresponding to the data.
        Patchsize (int): Size of the patch to be extracted.
        root (str): Path to the directory where the patches will be saved.
        Purpose (str, optional): Specifies whether the patches are for "training" or "testing".
                                 Defaults to "training".

    Returns:
        None: The extracted patches are saved to the specified directory.

    """

    Patches, Labels = Extract_Patches(Data, GT, Patchsize)
    store_path = join(root, (str(Patchsize) + "x" + str(Patchsize)))
    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    # Start labels from zero
    Labels = Labels - 1
    if Purpose == "training":
        # -- Save Patches and Labels when the scene is used as training scene
        np.save((store_path + "/Training_Patches.npy"), Patches)
        np.save((store_path + "/Training_Labels.npy"), Labels)
    elif Purpose == "test":
        np.save((store_path + "/Test_Patches.npy"), Patches)
        np.save((store_path + "/Test_Labels.npy"), Labels)

        # -- assume that Data is the test scene. To classify the test scene,it is necessary to extract
        # a patch around each pixel. By calling extracting_testscene_patches function, the patches are extracted
        test_scene_patches = extracting_testscene_patches(Data, Patchsize)
        # -- save
        torch.save(test_scene_patches, join(store_path, "test_scene_patches.pt"))


def extracting_testscene_patches(testscene: list, Patchsize: int) -> int:
    """
    Extracts patches around all pixels in a test scene, saves the patches, and prepares them
        for classification by a model.

        Args:
            testscene (list): ndarray representing the test scene data.
            Patchsize (int): Size of the patch to be extracted around each pixel.

        Returns:
            int: Patches extracted around each pixel, returned as a tensor.
    """

    # -- Pad size
    padd_size = Patchsize // 2
    if Patchsize % 2 == 0:  # even patch_size
        # (padd_size-1,padd_size) , ( padd_size-1,padd_size)  ,  (0,0)   ---> (top , bottom) , (left, right ), ( bands)
        Padding = ((padd_size - 1, padd_size), (padd_size - 1, padd_size), (0, 0))
    else:
        # odd patch_size
        Padding = ((padd_size, padd_size), (padd_size, padd_size), (0, 0))
    # -- Padding
    testscene = np.pad(testscene, pad_width=Padding, mode="edge")
    # -- Swap the channles
    testscene = np.swapaxes(testscene, 1, 2)
    testscene = np.swapaxes(testscene, 0, 1)
    testscene = np.float32(testscene)
    # -- Convert testscene into tensor
    testscene = np.expand_dims(testscene, axis=0)
    _, C, H, W = testscene.shape
    testscene = torch.tensor(testscene)
    # -- Extract a patche around each pixel
    test_scene_patches = view_as_windows_torch(
        image=testscene, shape=(Patchsize, Patchsize), stride=(1, 1)
    ).permute(0, 2, 3, 1, 4, 5)
    test_scene_patches = torch.squeeze(test_scene_patches)
    # -- return the patches
    return test_scene_patches


def Getting_Patches(
    DataDirectory: str,
    PatchSize: int,
    trainingscene="Quebec",
    testscene="Ottawa",
) -> None:
    """
    Extracts and saves patches from training and test scenes for classification.

        Args:
            DataDirectory (str): Path to the directory containing the data.
            PatchSize (int): Size of patches to be extracted around each ground truth sample.
            trainingscene (str, optional): Name of the training scene. Defaults to "Quebec".
            testscene (str, optional): Name of the test scene. Defaults to "Ottawa".

        Returns:
            None: Saves the extracted patches to the respective directories using the `saving_patches` function.

    """
    # -- Read labeled samples
    GT_training = scipy.io.loadmat(join(DataDirectory, trainingscene, "GT.mat"))["GT"]
    GT_testing = scipy.io.loadmat(join(DataDirectory, testscene, "GT.mat"))["GT"]

    print(f"\n training scene: {trainingscene}   test scene: {testscene}")

    training_root = join(
        DataDirectory,
        trainingscene,
        "Training",
    )
    testing_root = join(
        DataDirectory,
        testscene,
        "Testing",
    )

    CovarianceMatrix_Training = scipy.io.loadmat(
        join(training_root, "Standerized_Covariance_Matrix_dB.mat")
    )["Quebec_Standerized"]
    CovarianceMatrix_Testing = scipy.io.loadmat(
        join(testing_root, "Standerized_Covariance_Matrix_dB.mat")
    )["Ottawa_Standerized"]

    saving_patches(
        CovarianceMatrix_Training,
        GT_training,
        PatchSize,
        training_root,
        Purpose="training",
    )
    saving_patches(
        CovarianceMatrix_Testing,
        GT_testing,
        PatchSize,
        testing_root,
        Purpose="test",
    )
