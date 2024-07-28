"""
Reading Labeled Samples in each Scene and extract a patch around each one
"""
import scipy.io
import numpy as np
#from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
from scipy import io
import os
import torch


def Extract_Patches(Data: float, GT: int, Patchsize: int) -> [list, list]:
    """
    This code extract patches and corresponding labels around each ground truth sample

    Parameters
    ----------
    Data : float
        an array including the data.
    GT : int
        ground truth scene which has the same size as Data.
    Patchsize : int
        size of patch extracted around a labeled sample.

    Returns
    -------
    list
        array of patches.
    list
        array of Labels.

    """
    # -- Pad size
    padd_size = Patchsize//2
    if Patchsize % 2 == 0:  # even pad size
        Padding = ((padd_size - 1, padd_size),
                   (padd_size - 1, padd_size), (0, 0))
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

    # -- Padd Scene, Data, and GT
    Data = np.pad(Data, pad_width=Padding, mode='edge')
    GT = np.pad(GT, pad_width=Padding[0:2], mode='constant', constant_values=0)

    # -- Find indices of labeled pixels
    indices = np.transpose(np.nonzero(GT))

    # -- define labels and patch arrays for each labeled pixel
    Labels = np.zeros((indices.shape[0]))
    Patches = np.zeros(
        (indices.shape[0],  Patchsize, Patchsize, Data.shape[-1]))

    for i, loc in enumerate(indices):
        # -- extract a patch around a labeled sample
        patch = Data[loc[0] - Up_Row: loc[0] + Down_Row,
                     loc[1] - Left_Col: loc[1] + Right_Col, :]
        # -- save the patch into Patches array
        Patches[i, :, :, :] = patch
        # -- extract label of the sample
        Labels[i] = GT[loc[0], loc[1]]

    # -- Return patches along with their labels
    return np.float32(Patches), np.int8(Labels)


def view_as_windows_torch(image: float, shape: tuple, stride=None) -> list:
    """View tensor as overlapping rectangular windows, with a given stride.

    Parameters
    ----------
    image : `~torch.Tensor`
        4D image tensor, with the last two dimensions
        being the image dimensions
    shape : tuple of int
        Shape of the window.
    stride : tuple of int
        Stride of the windows. By default it is half of the window size.

    Returns
    -------
    windows : `~torch.Tensor`
        Tensor of overlapping windows

    """
    if stride is None:
        stride = shape[0] // 2, shape[1] // 2

    windows = image.unfold(2, shape[0], stride[0])
    return windows.unfold(3, shape[1], stride[1])


def saving_patches(Data: list,
                   GT: list,
                   Patchsize: int,
                   save_patch_root: str,
                   Purpose='training') -> None:
    """
    This function saves patches extracted around each ground truth samples.

    Parameters
    ----------
    Data : list
        an ndarray which can be covariance matrix or other features.
    GT : list
        ground truth scene.
    Patchsize : int
        patch size.
    save_patch_root : str
        where patches will be saved.
    Purpose : TYPE, optional
        Training and Test patches. The default is 'training'.

    Returns
    -------
    None
        Save the extracted patches.

    """

    Patches, Labels = Extract_Patches(Data, GT, Patchsize)
    store_path = join(save_patch_root, (str(Patchsize) + 'x' + str(Patchsize)))
    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    # Start labels from zero
    Labels = Labels-1
    if Purpose == 'training':
        # -- Save Patches and Labels when the scene is used as training scene
        np.save((store_path + '/Training_Patches.npy'), Patches)
        np.save((store_path + '/Training_Labels.npy'), Labels)
    elif Purpose == 'test':
        np.save((store_path + '/Test_Patches.npy'), Patches)
        np.save((store_path + '/Test_Labels.npy'), Labels)

        # -- assume that Data is the test scene. To classify the test scene,it is necessary to extract
        # a patch around each pixel. By calling extracting_testscene_patches function, the patches are extracted
        test_scene_patches = extracting_testscene_patches(Data, Patchsize)
        #-- save
        torch.save(test_scene_patches, join(
            store_path, 'test_scene_patches.pt'))


""" Note: this code shows how you can count occurrence of element in numpy array
count = np.count_nonzero(arr == 3)

"""

# this function extracts patches around each pixel


def extracting_testscene_patches(testscene: list,  Patchsize: int) -> int:
    """
    To classify the test scene, we need to extract a patch around all pixels 
    and save the patches.
    Then, the patches are feed into the model and their labels are specified.
    This function provides and saves the test scene patches.

    Parameters
    ----------
    testscene : list
        an ndarray list
    PatchSize : int
        size of a patch extracted around a pixel.

    Returns
    -------
    list
        patches extracted around each pixel


    """

    # -- Pad size
    padd_size = Patchsize//2
    if Patchsize % 2 == 0:   # even patch_size
        Padding = ((padd_size-1, padd_size), (padd_size-1, padd_size), (0, 0))
    else:  # odd patch_size
        Padding = ((padd_size, padd_size), (padd_size, padd_size), (0, 0))
    #-- Padding
    testscene = np.pad(testscene, pad_width=Padding, mode='edge')
    # -- Swap the channles
    testscene = np.swapaxes(testscene, 1, 2)
    testscene = np.swapaxes(testscene, 0, 1)
    testscene = np.float32(testscene)
    # -- Convert a test scene into tensor
    testscene = np.expand_dims(testscene, axis=0)
    _, C, H, W = testscene.shape
    testscene = torch.tensor(testscene)
    # -- Extract a patche around each pixel
    test_scene_patches = view_as_windows_torch(image=testscene, shape=(
        Patchsize, Patchsize), stride=(1, 1)).permute(0, 2, 3, 1, 4, 5)
    test_scene_patches = torch.squeeze(test_scene_patches)
    # -- return the patches
    return test_scene_patches


def Getting_Patches(DataDirectory: str,
                    PatchSize: int,
                    Features_name: str = 'Amplitude',
                    Standerized: str = 'yes') -> None:
    """


    Parameters
    ----------
    DataDirectory : str
        a path to data
    PatchSize : int
        size of patches extracted around each ground truth sample.
    Features_name : str
        which features should be used. The default is 'Amplitude'
    Standerized : str
        two different data can be used: with/without standerized dB.
        The default is 'yes'.

    Returns
    -------
    None
        save the patches using saving_patches function.

    """
    # -- Read labeled samples
    GT_training = scipy.io.loadmat(
        join(DataDirectory, 'Ottawa2', 'GT.mat'))['GT']
    GT_testing = scipy.io.loadmat(
        join(DataDirectory, 'Quebec3', 'GT.mat'))['GT']

    if Standerized == 'no':

        training_root = join(DataDirectory, 'Ottawa2',
                             'Features', Features_name, 'dB')
        testing_root = join(DataDirectory, 'Quebec3',
                            'Features', Features_name, 'dB')

        CovarianceMatrix_Training = scipy.io.loadmat(join(
            training_root, 'Covariance_Matrix_dB.mat'))[

            'Covariance_Matrix_dB']  # [C11, C12, C22]
        CovarianceMatrix_Testing = scipy.io.loadmat(join(
            testing_root, 'Covariance_Matrix_dB.mat'))[
            'Covariance_Matrix_dB']  # [C11, C12, C22]

        saving_patches(CovarianceMatrix_Training, GT_training,
                       PatchSize, training_root, Purpose='training')
        saving_patches(CovarianceMatrix_Testing, GT_testing,
                       PatchSize, testing_root, Purpose='test')

    else:

        training_root = join(DataDirectory, 'Ottawa2', 'Features',
                             Features_name, 'Standerized dB', 'Training')

        testing_root = join(DataDirectory, 'Quebec3', 'Features',
                            Features_name, 'Standerized dB', 'Testing')

        CovarianceMatrix_Training = scipy.io.loadmat(join(
            training_root, 'Standerized_Covariance_Matrix_dB.mat'))
        ['Ottawa_Standerized']
        CovarianceMatrix_Testing = scipy.io.loadmat(
            join(testing_root, 'Standerized_Covariance_Matrix_dB.mat'))
        ['Quebec_Standerized']

        saving_patches(CovarianceMatrix_Training, GT_training,
                       PatchSize, training_root, Purpose='training')
        saving_patches(CovarianceMatrix_Testing, GT_testing,
                       PatchSize, testing_root, Purpose='test')
