import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class TransformsToolkit(object):

    @staticmethod
    def transform_bigearthnet_train(image_size):
        """Trying to assemble a transform that makes sense for satellite images"""
        return transforms.Compose([
            # Rotate the image by angle
            transforms.RandomRotation((0, 360), interpolation=InterpolationMode.BILINEAR),
            # Flip the given image randomly with a given probability
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # Blurs image with randomly chosen Gaussian blur
            transforms.GaussianBlur(kernel_size=13)  # default for sigma is: sigma=(0.1, 2.0)
        ])

    @staticmethod
    def transform_bigearthnet_eval(image_size):
        """Same as above: trying to make sense for satellite images"""
        return transforms.Compose([
        ])

    @staticmethod
    def transform_original_simclr(image_size):
        """This is the data augmentation transform described in the SimCLR paper"""
        return transforms.Compose([
            # Crop a random portion of image and resize it to a given size
            transforms.RandomResizedCrop(image_size),
            # Flip the given image randomly with a given probability
            transforms.RandomHorizontalFlip(p=0.5),
            # Apply randomly a list of transformations with a given probability
            transforms.RandomApply(nn.ModuleList([
                # Randomly change the brightness, contrast, saturation and hue of an image
                transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            ]), p=0.8),
            # Randomly convert image to grayscale with a probability of p
            transforms.RandomGrayscale(p=0.2),
            # Blurs image with randomly chosen Gaussian blur
            transforms.GaussianBlur(kernel_size=int(0.1 * image_size), sigma=(0.1, 2.0)),
        ])
