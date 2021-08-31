import json
import os

import torch
import pickle
import sys
from torch import optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


sys.path.extend(['./', './yolact_package'])

from fashionadv_utils.util import normalize, undo_normalize, get_item, get_image_to_attack
from fashionadv_utils.textural_loss import CrossGramMatrix, content_layers, style_layers, vgg
from fashionadv_utils.fashionAdv_attach import optimze_attack

from yolact_package.eval import prep_coco_cats
from yolact_package.utils.augmentations import NoneTransform
from yolact_package.yolact import Yolact
from yolact_package.layers.output_utils import postprocess
from yolact_package.data import COCODetection


# ---------------- Configuration ----------------
with open('./configuration.json', 'r') as f:
    configuration = json.load(f)


# ---------------- Configuration Torch ----------------
device = torch.device("cuda")
cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ---------------- Setup ----------------
attack_setup = configuration['attack_setup']

dataset = COCODetection(f"{configuration['val2017']}/", f"{configuration['annotations']}/instances_val2017.json",
                        transform=NoneTransform(), has_gt=True)
prep_coco_cats()

yolact = Yolact()
yolact.to(device)
yolact.load_weights(f"{configuration['weights_yolact']}")
yolact.eval()

yolact.detect.use_fast_nms = True
yolact.detect.use_cross_class_nms = False
yolact.change_pred_outs(False)

for param in yolact.parameters():
    param.requires_grad = False


def fashionAdv_attack(content_index_image, setup):
    # Getting the clean image from dataset.
    content_image, _, _, num_crowds = get_item(content_index_image, dataset)
    if content_index_image == 4011:
        num_crowds = 0
    content_image = content_image.to(device)

    # Generating the ground truth of YOLACT
    yolact.change_pred_outs(False)
    y_hat = yolact(content_image)
    nb_keep = max(1, sum(y_hat[0]['detection']['score'] > setup['threshold_yolact']))
    ind = (y_hat[0]['detection']['class'] != 0).nonzero()[:nb_keep]
    classe = y_hat[0]['detection']['class'][ind]
    box = y_hat[0]['detection']['box'][ind].squeeze(1)
    targets_image = torch.cat([box, classe.float()], dim=1).to(device)
    masks_image = postprocess(y_hat, 550, 550, crop_masks=True, score_threshold=0)[3][ind].squeeze(1).to(device)

    # Open the mask for the attack, generating from the human parsing
    image_mask = Image.open(f'{configuration["mask_upper_shirt"]}/mask_{dataset.ids[content_index_image]:012d}.png')
    content_mask = transforms.ToTensor()(image_mask).to(device)

    # Create the patch
    patch = torch.randn(content_image.size()).type_as(content_image.data).to(device)
    patch.data.copy_(content_image)
    patch = Variable((content_mask * patch), requires_grad=True)
    optimizer = optim.Adam([patch], lr=setup['lr'], amsgrad=True)

    # Open the style image
    im = Image.open(f'{configuration["fashion_pattern"]}/{setup["Texture_style"]:0>2d}.jpg')
    style_image = transforms.ToTensor()(im)
    style_image = normalize(style_image).cuda()

    style_weights = [setup['textural_loss_weight'] / n ** 2 for n in [64, 128, 256, 512, 512]]
    setup['weights'] = style_weights + [0]
    Crosstyle_t = [CrossGramMatrix()(A, B).detach() for A, B in
                         zip(vgg(style_image, style_layers[:4]), vgg(style_image, style_layers[1:]))]
    content_t = [A.detach() for A in vgg(content_image, content_layers)]

    for iteration in range(setup['max_iter']):
        yolact.change_pred_outs(True)
        optimze_attack(yolact, optimizer, patch, content_mask, content_image, targets_image, masks_image, num_crowds, Crosstyle_t, content_t, setup)
        patch.data = normalize(torch.clamp(undo_normalize(patch.clone()), 0, 1))  # Keep the patch between 0 and 1

    return patch


if __name__ == '__main__':
    if len(sys.argv) == 3:
        id_first_image = int(sys.argv[1])
        id_last_image = int(sys.argv[2])
    else:
        id_first_image = 0
        id_last_image = 1000

    imgs_to_attack = get_image_to_attack(dataset, configuration['image_id_list'])

    with open(configuration['minimal_texture_cost'], 'rb') as file_min_cost:
        style_id = pickle.load(file_min_cost)

    if configuration['result'] not in os.listdir():
        os.mkdir(configuration['result'])

    for i in tqdm(range(id_first_image, id_last_image)):
        attack_setup['Texture_style'] = style_id[i]
        img_adv = fashionAdv_attack(imgs_to_attack[i], attack_setup)[0]

        img_adv_undo = undo_normalize(img_adv)
        save_image(img_adv_undo, f"{configuration['result']}/patch_{dataset.ids[imgs_to_attack[i]]:012d}.png")
