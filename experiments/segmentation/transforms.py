import torch
from torchvision.transforms import *

import albumentations as albu
import torchvision.transforms as transform


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=0.0, p=0.5, border_mode=0),
        # albu.PadIfNeeded(min_height=1000, min_width=1600, always_apply=True, border_mode=0),
        albu.RandomCrop(height=256, width=400, always_apply=True),
        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),
        #
        # albu.OneOf(
        #     [
        #         albu.CLAHE(),
        #         albu.RandomBrightnessContrast(),
        #         albu.HueSaturationValue(),
        #         albu.RandomGamma(),
        #     ],
        #     p=0.9,
        # ),
        # #
        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.CLAHE(),
        # albu.RandomBrightnessContrast(),
        # albu.HueSaturationValue(),
        albu.Normalize(),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Normalize(),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(test_transform)


def get_inference_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Normalize(),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    # TODO: UserWarning: Using lambda is incompatible with multiprocessing. Consider using regular functions or partial().
    #   warnings.warn('Using lambda is incompatible with multiprocessing.
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
