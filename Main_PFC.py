# %% Libraries

import os
import torch.nn as nn
import time
import torch
import numpy as np
import argparse

from sklearn.metrics import accuracy_score
from Generating_Patches import Getting_Patches
from PFC import pfc_classifier
from torch import optim as optim
from util import train_model, confusionmatrix, SceneClassification, creating_dataloaders


def get_parameters():
    parser = argparse.ArgumentParser(description='Land Cover Classification')
    parser.add_argument('--file_dir',
                        type=str,
                        default='D:\My Projects\Land_Cover_Classification',
                        help='Base directory of the project')
    parser.add_argument('--data_dir',
                        type=str,
                        default=None,
                        help='Data directory')
    parser.add_argument('--result_dir',
                        type=str,
                        default=None,
                        help='Result directory')
    parser.add_argument('--num_classes',
                        type=int,
                        default=5,
                        help='Number of classes')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help='Learning rate')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs')

    # Add pfc_config arguments
    parser.add_argument('--img_size',
                        type=int,
                        default=32,
                        help='Image size')
    parser.add_argument('--patch_size',
                        type=int,
                        default=1,
                        help='Patch size')
    parser.add_argument('--in_chans',
                        type=int,
                        default=3,
                        help='Number of input channels')
    parser.add_argument('--apply_emd',
                        type=bool,
                        default=True,
                        help='Apply EMD')
    parser.add_argument('--embed_dim',
                        type=int,
                        default=16,
                        help='Embedding dimension')
    parser.add_argument('--out_channels',
                        nargs='+',
                        type=int,
                        default=[32, 64, 128, 128],
                        help='Output channels')
    parser.add_argument('--pyramid_fusion',
                        type=bool,
                        default=True,
                        help='Pyramid Fusion')
    parser.add_argument('--depths',
                        nargs='+',
                        type=int,
                        default=[2, 2, 2, 2],
                        help='Depths')
    parser.add_argument('--num_heads',
                        nargs='+',
                        type=int,
                        default=[1, 4, 4, 8],
                        help='Number of heads')
    parser.add_argument('--window_size',
                        type=int,
                        default=4,
                        help='Window size')
    parser.add_argument('--mlp_ratio',
                        type=float,
                        default=4.0,
                        help='MLP ratio')
    parser.add_argument('--qkv_bias',
                        type=bool,
                        default=True,
                        help='QKV bias')
    parser.add_argument('--drop_path_rate',
                        type=float,
                        default=0.2,
                        help='Drop path rate')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parameters()
    FileDirectory = r'D:\My Projects\Land_Cover_Classification'  # n
    DataDirectory = args.data_dir if args.data_dir else os.path.join(
        args.file_dir, 'Data')
    Result_Directory = args.result_dir if args.result_dir else os.path.join(FileDirectory,
                                                                            'results of all models',
                                                                            'Training on Ottawa and Test on Quebec',
                                                                            'PFC')

    if os.path.exists(Result_Directory) is not True:
        os.mkdir(Result_Directory)

    # -- filter out only the folder names
    AllScenes = [item for item in os.listdir(
        DataDirectory) if os.path.isdir(os.path.join(DataDirectory, item))]

    # -- config of glocal
    pfc_config = {"img_size": args.img_size,
                  "patch_size": args.patch_size,
                  "in_chans": args.in_chans,
                  "apply_emd": args.apply_emd,
                  "embed_dim": args.embed_dim,
                  "out_channles": args.out_channels,
                  "pyramid_Fusion": args.pyramid_fusion,
                  "num_classes": args.num_classes,
                  "depths": args.depths,
                  "num_heads": args.num_heads,
                  "window_size": args.window_size,
                  "mlp_ratio": args.mlp_ratio,
                  "qkv_bias": args.qkv_bias,
                  "drop_path_rate": args.drop_path_rate}

    score_funcs = {'Accuracy': accuracy_score}
    # -- Create training, validating, and testing pathces
    Getting_Patches(DataDirectory=DataDirectory,
                    PatchSize=args.img_size,
                    Features_name='Amplitude',
                    Standerized='yes')

    # -- Measure perfromance of a model using different percentage of using samples
    for train_percentage in np.arange(10, 90, 10):
        print(f'{train_percentage}% of training samples')
        # -- Creat a path to save the result
        result_path = os.path.join(Result_Directory,
                                   (str(train_percentage) +
                                    'percentage training sampels'))
        # -- make sure the result_path is available
        if os.path.exists(result_path) is not True:
            os.mkdir(result_path)

        model = pfc_classifier(**pfc_config)
        new_model = pfc_classifier(**pfc_config)

        # -- define the optimizer
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=0.05
                                )
        # create loaders
        Train_Loader, Val_Loader, Test_Loader = creating_dataloaders(
            DataDirectory,
            'Ottawa2',
            'Quebec3',
            train_percentage,
            pfc_config['img_size'],
            args.batch_size
        )

        Start = time.time()

        # -- Train the model
        train_model(args.epochs,
                    model,
                    optimizer,
                    Train_Loader,
                    nn.CrossEntropyLoss(),
                    score_funcs,
                    result_path,
                    pfc_config['img_size'],
                    validation_loader=Val_Loader,
                    test_loader=Test_Loader)

        End = time.time()
        Diff_min = (End-Start)/3600
        print('***********      End of Training        **************')
        print(f'\n Test Quebec Scene has been done.')
        print('\n It tooke: {:.3f} hours'.format(Diff_min))
        print('\n Scene Classification starts \n')

        # -- Measure confusion matrix

        confusionmatrix(Test_Loader, model, result_path,
                        pfc_config['img_size'])

        # -- Scene Classification

        TestScene = torch.load(os.path.join(DataDirectory,
                                            'Quebec3',
                                            'Features',
                                            'Amplitude',
                                            'Standerized dB',
                                            'Testing',
                                            (str(
                                                pfc_config['img_size'])+'x'+str(pfc_config['img_size'])),
                                            'test_scene_patches.pt'))

        SceneClassification(TestScene,
                            new_model,
                            result_path,
                            pfc_config['img_size'],
                            num_classes=args.num_classes,
                            Row_Step=None,
                            patchratio=100,
                            LandMask=None
                            )
