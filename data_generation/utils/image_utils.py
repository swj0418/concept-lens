import numpy as np

def convert_images_to_uint8(images, drange=[-1, 1], nchw_to_nhwc=False):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    if nchw_to_nhwc:
        images = np.transpose(images, [0, 2, 3, 1])

    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    np.clip(images, 0, 255, out=images)
    images = images.astype('uint8')
    return images


def convert_images_from_uint8(images, drange=[-1, 1], nhwc_to_nchw=False):
    """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
    Can be used as an input transformation for Network.run().
    """
    if nhwc_to_nchw:
        images = np.rollaxis(images, 3, 1)
    return images / 255 * (drange[1] - drange[0]) + drange[0]