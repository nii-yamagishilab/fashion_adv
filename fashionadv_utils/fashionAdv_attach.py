import numpy
import os
from pytorch_msssim import ms_ssim
import random
import torch

from fashionadv_utils.jpeg_compression import jpeg_approximation
from fashionadv_utils.image_manipulation import apply_gaussian_filter, adjust_hue
from fashionadv_utils.textural_loss import CrossGramMatrix, content_layers, loss_fns, style_layers, vgg
from fashionadv_utils.util import normalize, undo_normalize
import fashionadv_utils.functional_tensor as ft

from kornia import kornia
from kornia.augmentation import random_generator as rg
from kornia.losses import TotalVariation
from yolact_package.data import cfg, set_cfg
from yolact_package.eval import parse_args
from yolact_package.layers.modules import MultiBoxLoss

seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


device = torch.device("cuda")

set_cfg("yolact_base_config")

args = parse_args('')
args.top_k = 5
cfg.mask_proto_debug = args.mask_proto_debug
cfg.train_masks = True
cfg.use_semantic_segmentation_loss = True
cfg.mask_proto_loss = 'l1'

criterion = MultiBoxLoss(num_classes=cfg.num_classes, pos_threshold=cfg.positive_iou_threshold,
                         neg_threshold=cfg.negative_iou_threshold, negpos_ratio=cfg.ohem_negpos_ratio)
TV = TotalVariation()


def optimze_attack(yolact, optim, img_to_optimize, mask, real_image, target, masks, num_crowd, Crosstyle_texture,
                   content_texture, option):

    optim.zero_grad()
    adv_x = torch.mul((1 - mask), real_image) + torch.mul(mask, img_to_optimize)

    # TV loss
    tv_loss = option['tv_loss_weight'] * TV(adv_x)
    total_loss = tv_loss

    # SSIM loss
    ssim_loss = option['ssim_loss_weight'] * ms_ssim(undo_normalize(real_image), undo_normalize(adv_x), data_range=1,
                                                     size_average=True)
    total_loss -= ssim_loss

    # Texture loss
    Cross_out = zip(vgg(adv_x, style_layers[:4]), vgg(adv_x, style_layers[1:]))
    content_out = vgg(adv_x, content_layers)
    layer_losses = [option['weights'][a] * loss_fns[a](A, B, Crosstyle_texture[a]) / CrossGramMatrix()(A, B).std()
                    for a, (A, B) in enumerate(Cross_out)]
    content_losses = [option['weights'][4 + a] * loss_fns[4 + a](A, content_texture[a]) for a, A in
                      enumerate(content_out)]
    texture_loss = sum(layer_losses) + sum(content_losses)
    total_loss += texture_loss

    # Transform operation
    if random.random() > (1 - option['apply_transform']):
        params = rg.random_perspective_generator(1, 550, 550, 1.0, option['transform'])
        adv_x = undo_normalize(adv_x)

        adv_x = normalize(kornia.apply_perspective(adv_x, params))
        masks = kornia.apply_perspective(masks, params).to(device).squeeze(0)

    if random.random() > (1 - option['apply_gaussian']):  # apply Gaussian Blur
        adv_x = undo_normalize(adv_x)
        adv_x = normalize(apply_gaussian_filter(adv_x, option))

    if random.random() > (1 - option['apply_color_manipulation']):
        adv_x = undo_normalize(adv_x).squeeze(0)
        adv_x = ft.adjust_brightness(adv_x, random.uniform(*option['brightness']))
        adv_x = ft.adjust_saturation(adv_x, random.uniform(*option['saturation']))
        adv_x = ft.adjust_contrast(adv_x, random.uniform(*option['contrast']))
        adv_x = adjust_hue(adv_x, random.uniform(*option['hue']))
        noise = torch.cuda.FloatTensor(adv_x[0].shape).uniform_(-1, 1) * random.uniform(*option['noise'])
        adv_x = normalize((adv_x + noise).unsqueeze(0))

    if random.random() > (1 - option['apply_jpeg']):
        adv_x = undo_normalize(adv_x).squeeze(0)
        adv_x = torch.clamp(adv_x, 0, 1) * 255

        qf = random.uniform(*option['quality_factor'])
        adv_x = jpeg_approximation(adv_x.permute(1, 2, 0).unsqueeze(0), factor=qf) / 255
        adv_x = adv_x.permute(0, 3, 1, 2)
        adv_x = normalize(adv_x).float()

    # Adversarial Loss
    yolact.change_pred_outs(True)
    yolact_loss = criterion(yolact, yolact(adv_x), [target], [masks], [num_crowd])

    adv_loss = option['adv_loss_weight'] * (yolact_loss['S'] * option['yolact_weights']['segmentation_weight'] +
                                            yolact_loss['C'] * option['yolact_weights']['class_weight'])

    total_loss += adv_loss
    total_loss.backward()
    optim.step()
