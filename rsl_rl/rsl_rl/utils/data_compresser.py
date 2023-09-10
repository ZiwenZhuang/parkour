import numpy as np

def compress_normalized_image(images):
    """ Compress the 0~1 (float32) image to 0~255 (uint8) might reduce accuracy """
    images = np.clip(images, 0, 1)
    return (images * 255).astype(np.uint8)

def decompress_normalized_image(images):
    """ Decompress the 0~255 (uint8) image to 0~1 (float32) """
    return images.astype(np.float32) / 255
