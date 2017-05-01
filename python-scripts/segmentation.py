# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on
top of the Caffe deep learning library.

Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)

Supervisor:
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""

import sys
import time
import getopt
import os
import numpy as np
from PIL import Image as PILImage

# Path of the Caffe installation.
_CAFFE_ROOT = "../caffe/"

# Model definition and model file paths
_MODEL_DEF_FILE = "TVG_CRFRNN_new_deploy.prototxt"  # Contains the network definition
_MODEL_FILE = "TVG_CRFRNN_COCO_VOC.caffemodel"  # Contains the trained weights. Download from http://goo.gl/j7PrPZ

_GPU_DIVICE = 0

_INPUT_PATH = '2016/'
_OUTPUT_PATH = '2016_output/'


sys.path.insert(0, _CAFFE_ROOT + "python")
import caffe

_MAX_DIM = 400


def load_pretained_net(model_def_file, model_file, gpu_device):
    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)
    return net


def crfrnn_segmenter(net, inputs):
    """ Returns the segmentation of the given image.

    Args:
        model_def_file: File path of the Caffe model definition prototxt file
        model_file: File path of the trained model file (contains trained weights)
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
        inputs: List of images to be segmented

    Returns:
        The segmented image
    """

    # assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    # assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
    #                                     "./download_trained_model.sh").format(model_file)

    # if gpu_device >= 0:
    #     caffe.set_device(gpu_device)
    #     caffe.set_mode_gpu()
    # else:
    #     caffe.set_mode_cpu()

    # net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."

    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)


def run_crfrnn(input_file, output_file, gpu_device, net):
    """ Runs the CRF-RNN segmentation on the given RGB image and saves the segmentation mask.

    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    # palette = get_palette(256)

    mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    segmentation = crfrnn_segmenter(net, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    # output_im = PILImage.fromarray(segmentation)
    # output_im.putpalette(palette)
    # output_im.save(output_file)

    mask = np.where(segmentation==15, 1, 0).astype('uint8')
    output_image = image * mask[:, :, np.newaxis]
    output_image = PILImage.fromarray(np.uint8(output_image))
    output_image.save(output_file)


def resize_image(image):
    """ Resizes the image so that the largest dimension is not larger than 500 pixels.
        If the image's largest dimension is already less than 500, no changes are made.

    Args:
        Input image

    Returns:
        Resized image where the largest dimension is less than 500 pixels
    """

    width, height = image.shape[0], image.shape[1]
    max_dim = max(width, height)

    if max_dim > _MAX_DIM:
        if height > width:
            ratio = float(_MAX_DIM) / height
        else:
            ratio = float(_MAX_DIM) / width
        image = PILImage.fromarray(np.uint8(image))
        image = image.resize((int(height * ratio), int(width * ratio)), resample=PILImage.BILINEAR)
        image = np.array(image)

    return image


def main():
    """ Main entry point to the program. """

    # input_file = "input.jpg"
    # output_file = "output.jpg"
    gpu_device = _GPU_DIVICE  # Use -1 to run only on the CPU, use 0-3[7] to run on the GPU

    net = load_pretained_net(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device)
    imgs = os.listdir(_INPUT_PATH)
    for img in imgs:
        run_crfrnn(_INPUT_PATH + img, _OUTPUT_PATH + img, gpu_device, net)


if __name__ == "__main__":
    main()
