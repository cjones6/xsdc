import numpy as np
import torch
import torchvision.transforms.functional as TF

from src import default_params as defaults


def random_rotation(x):
    """
    Apply a random rotation to an image x with probability 0.5. If a rotation is applied, it is chosen to lie between
    -30 degrees and 30 degrees.

    :param x: Image to potentially be rotated
    :return: x: The (potentially) rotated image
    """
    rotate = np.random.choice(range(2), size=1)
    if rotate:
        angle = 30*np.clip(np.random.normal(0, 0.33), a_max=1, a_min=-1)
        x = TF.rotate(TF.to_pil_image(x.cpu()), angle)
    else:
        x = TF.to_pil_image(x.cpu())

    return x


def random_width(x, orig_width, p=0.1):
    """
    Apply a randomly compression of the width of the image with probability p. The random width is chosen to be at
    most 1/4th of the width of the original image. The image is then zero padded to be the same size as the original
    image.

    :param x: Image to potentially be compressed
    :param orig_width: Original width of the image
    :param p: Probability with which the compression is applied
    :return: The (potentially) compressed and then padded image
    """
    if np.random.choice(range(2), size=1, p=(1-p, p)):
        width = orig_width - int(np.floor(orig_width/4*np.clip(np.abs(np.random.normal(0, 0.33)), a_max=1, a_min=-1)+1))
        x = TF.resize(x, [orig_width, width])
        if width % 2 == 0:
            x = TF.pad(x, ((orig_width-width)//2, 0, (orig_width-width)//2, 0))
        else:
            if np.random.choice(range(2), size=1):
                x = TF.pad(x, ((orig_width-width)//2, 0, (orig_width-width)//2 + 1, 0))
            else:
                x = TF.pad(x, ((orig_width-width)//2 + 1, 0, (orig_width-width)//2, 0))

    return np.array(x)[np.newaxis, :, :]


def random_shift(x, p=0.25):
    """
    Apply a random shift to the digit in the image with probability 2p.

    :param x: Image to potentially be shifted
    :param p: Half of the probability of shifting the image
    :return: x: The (potentially) shifted image
    """
    shift_left_right = np.random.choice(range(2), size=1, p=(1-p, p))
    shift_up_down = np.random.choice(range(2), size=1, p=(1-p, p))
    if shift_left_right:
        nonzero_cols = np.where(np.sum(x, 2) > 0)[1]
        x_translation = np.clip(np.random.normal(0, 0.33), a_min=-nonzero_cols[0], a_max=x.shape[1]-nonzero_cols[-1])
        if 1 > x_translation >= 0:
            x_translation += 1
        elif -1 < x_translation < 0:
            x_translation -= 1
        x_translation = int(x_translation)
        x = np.roll(x, x_translation, axis=2)

    if shift_up_down:
        nonzero_cols = np.where(np.sum(x, 1) > 0)[1]
        y_translation = np.clip(np.random.normal(0, 0.33), a_min=-nonzero_cols[0], a_max=x.shape[1]-nonzero_cols[-1])
        if 1 > y_translation >= 0:
            y_translation += 1
        elif -1 < y_translation < 0:
            y_translation -= 1
        y_translation = int(y_translation)
        x = np.roll(x, y_translation, axis=1)

    return x


def random_erasure(x):
    """
    Randomly set to zero a 4x4 section of the image.
    :param x: Image in which a part will be zeroed
    :return: x: Image with a part that has been zeroed
    """
    rand_amts = np.random.uniform(size=2)
    mask_x = int(np.floor(rand_amts[0] * 19))
    mask_y = int(np.floor(rand_amts[1] * 19))
    x[:, mask_y-2:mask_y+2, mask_x-2:mask_x+2] = 0
    return x


def augment(x, y, y_unlabeled_truth=None, factor=10):
    """
    Perform data augmentation as in the following paper:

    - Byerly A, Kalganova T, Dear I (2020) A branching and merging convolutional network with homogeneous filter capsules.
    CoRR abs/2001.09136
    :param x: Tensor with the images to be augmented
    :param y: Tensor with the labels for the images
    :param y_unlabeled_truth: Tensor with the true labels for images whose labels are unknown to the algorithm
    :param factor: Number of augmentations to perform for each image
    :return: A tuple containing a subset of:

        * torch.stack(new_xs): The augmented images
        * torch.LongTensor(new_ys): The labels of the augmented images (when the labels are known to the algorithm)
        * torch.LongTensor(new_y_unlabeled_truths): The labels of the augmented images (when the labels are unknown to
                                                    the algorithm)
    """
    orig_width = x.shape[-1]
    new_xs = []
    new_ys = []
    new_y_unlabeled_truths = []
    for i in range(len(x)):
        for j in range(factor):
            xi = random_rotation(x[i].type(torch.IntTensor))
            xi = random_width(xi, orig_width)
            xi = random_shift(xi)
            xi = random_erasure(xi)

            new_xs.append(torch.from_numpy(xi))
            new_ys.append(y[i])
            if y_unlabeled_truth is not None:
                new_y_unlabeled_truths.append(y_unlabeled_truth[i])

    if y_unlabeled_truth is not None:
        return torch.stack(new_xs), torch.LongTensor(new_ys), torch.LongTensor(new_y_unlabeled_truths)
    else:
        return torch.stack(new_xs), torch.LongTensor(new_ys), None
