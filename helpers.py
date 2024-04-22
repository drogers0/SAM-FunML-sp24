import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision
from config import *
from PIL import Image



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=GREEN_COLOR, marker='*', s=marker_size, edgecolor='white',
            linewidth=LINEWIDTH)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color=RED_COLOR, marker='*', s=marker_size, edgecolor='white',
            linewidth=LINEWIDTH)

def closetn(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def downsample(image, ground_truth=False):
    if ground_truth:
        image = cv2.cvtColor((np.array(((image + 1) / 2) * 255, dtype='uint8')), cv2.COLOR_GRAY2RGB)
    height, width = image.shape[:2] #width, height of the original image

    downsample_image = cv2.pyrDown(image, dstsize=(width // 2, height // 2)) # cv2 downsample function

    # Zero-pad downsampled image
    padded_image = torchvision.transforms.Pad((width//4, height//4, width//4, height//4)) # left, top, right, bottom
    result = padded_image(Image.fromarray(downsample_image))
    result = result.resize((width, height)) #resize to the original image
    result = np.asarray(result)
    if ground_truth:
        result = result[:,:,0] == 255
    return result