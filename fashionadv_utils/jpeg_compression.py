import torch
import itertools
import numpy as np

y_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32).T
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47],
                            [18, 21, 26, 66],
                            [24, 26, 56, 99],
                            [47, 66, 99, 99]]).T


def rgb_to_ycbcr_jpeg(image):
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]], dtype=np.float32).T
    shift = [0., 128., 128.]

    result = torch.tensordot(image, torch.tensor(matrix), dims=1) + torch.tensor(shift)
    return result


def downsampling_420(image):
    #  image: [H, W, 3]
    # output: tuple of length 3
    #   y:  height x width
    #   cb: height/2 x width/2
    #   cr: height/2 x width/2
    y, cb, cr = torch.split(image, 1, 3)

    avg2d = torch.nn.AvgPool2d((2, 2), (2, 2), 0)
    cb = avg2d(cb.permute(0, 3, 1, 2))
    cr = avg2d(cr.permute(0, 3, 1, 2))
    return y.squeeze(-1), cb.squeeze(0), cr.squeeze(0)


def image_to_patches(image):
    # input: h x w
    # output: h*w/64 x 8 x 8
    k = 8
    batch_size, height, width = image.shape
    image_reshaped = torch.reshape(image, (batch_size, height // k, k, -1, k))
    image_transposed = image_reshaped.permute((0, 1, 3, 2, 4))
    return torch.reshape(image_transposed, (batch_size, -1, k, k))


def dct_8x8(image):
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    result = torch.tensor(scale) * torch.tensordot(image, torch.tensor(tensor), dims=2)
    return result


def y_quantize(image, rounding, factor=1):
    image = image / (torch.tensor(y_table * factor))
    image = rounding(image)
    return image


def c_quantize(image, rounding, factor=1):
    image = image / (torch.tensor(c_table * factor))
    image = rounding(image)
    return image


def y_dequantize(image, factor=1):
    return image * (torch.tensor(y_table * factor))


def c_dequantize(image, factor=1):
    return image * (torch.tensor(c_table * factor))


def idct_8x8(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * torch.tensor(alpha)

    tensor = np.zeros((8, 8, 8, 8))  # , dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)

    result = 0.25 * torch.tensordot(image, torch.tensor(tensor), dims=2) + 128
    return result


def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    k = 8
    batch_size = patches.shape[0]
    image_reshaped = torch.reshape(patches, (int(batch_size), int(height // k), int(width // k), int(k), int(k)))
    image_transposed = image_reshaped.permute((0, 1, 3, 2, 4))
    return torch.reshape(image_transposed, (int(batch_size), int(height), int(width)))


def upsampling_420(y, cb, cr):
    # input:
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    # output:
    #   image: batch x height x width x 3
    def repeat(x, k=2):
        height, width = x.shape[1:3]
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, k, k)
        x = torch.reshape(x, (-1, height * k, width * k))
        return x

    cb = repeat(cb)
    cr = repeat(cr)
    return torch.stack((y, cb, cr), axis=-1)


def ycbcr_to_rgb_jpeg(image):
    matrix = np.array([[1., 0., 1.402],
                       [1, -0.344136, -0.714136],
                       [1, 1.772, 0]]).T
    shift = [0, -128, -128]

    result = torch.tensordot((image + torch.tensor(shift)), torch.tensor(matrix), dims=1)
    return result


def diff_round(x):
    return torch.round(x) + (x - torch.round(x)) ** 3


def round_only_at_0(x):
    cond = (torch.abs(x) < 0.5).type(torch.cuda.FloatTensor)
    return cond * (x ** 3) + (1 - cond) * x


def jpeg_approximation(image, rounding=round_only_at_0, factor=1):
    """
    image: [B, H, W, C]
    """
    height, width = image.shape[1:3]

    orig_height, orig_width = height, width
    if height % 16 != 0 or width % 16 != 0:
        # Round up to next multiple of 16
        height = ((height - 1) // 16 + 1) * 16
        width = ((width - 1) // 16 + 1) * 16

        vpad = height - orig_height
        wpad = width - orig_width
        top = vpad // 2
        bottom = vpad - top
        left = wpad // 2
        right = wpad - left

        image = torch.nn.functional.pad(image, (0, 0, top, bottom, left, right, 0, 0))

    # "Compression"
    image = rgb_to_ycbcr_jpeg(image)
    y, cb, cr = downsampling_420(image)

    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = components[k]
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)
        comp = c_quantize(comp, rounding, factor) if k in ('cb', 'cr') else y_quantize(comp, rounding, factor)
        components[k] = comp

    for k in components.keys():
        comp = components[k]
        comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(comp, factor)
        comp = idct_8x8(comp)

        if k in ('cb', 'cr'):
            comp = patches_to_image(comp, height / 2, width / 2)
        else:
            comp = patches_to_image(comp, height, width)
        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']

    image = upsampling_420(y, cb, cr)
    image = ycbcr_to_rgb_jpeg(image)

    # Crop to original size
    if orig_height != height or orig_width != width:
        image = image[:, top:-bottom, left:-right]

    image = torch.clamp(image, 0, 255)
    return image
