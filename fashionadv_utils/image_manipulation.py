import math
import random
import torch
import torch.nn.functional as F


def apply_gaussian_filter(image, option, size=550):
    # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
    kernel_size = random.choice(option['kernel_size'])
    sigma = random.uniform(*option['sigma'])
    channels = 3
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return F.interpolate(gaussian_filter(image), size=size)


"""
This come from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#adjust_hue
"""


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (Tensor): Image to be adjusted. Image type is either uint8 or float.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
         Tensor: Hue adjusted image.
    """
    img = _rgb2hsv(img)
    # h, s, v = img.unbind(0)
    img[0] += hue_factor
    img[0] = img[0] % 1.0
    # img = torch.stack((h, s, v))
    img_hue_adj = _hsv2rgb(img)

    return img_hue_adj


def _rgb2hsv(img):
    r, g, b = img.unbind(0)

    maxc = torch.max(img, dim=0).values
    minc = torch.min(img, dim=0).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc))


def _hsv2rgb(img):
    h, s, v = img.unbind(0)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i == torch.arange(6)[:, None, None]

    a1 = torch.stack((v, q, p, p, t, v))
    a2 = torch.stack((t, v, v, q, p, p))
    a3 = torch.stack((p, p, t, v, v, q))
    a4 = torch.stack((a1, a2, a3))

    return torch.einsum("ijk, xijk -> xjk", mask.to(dtype=img.dtype), a4)
