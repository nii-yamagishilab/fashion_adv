import os
import torch
import numpy as np
import random
from tqdm import tqdm
import time
import itertools
import shutil


import pickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn.functional import interpolate

import sys
from pytorch_msssim import ssim

from fashionadv_utils.util import normalize_imagenet, get_item

sys.path.extend(['./', './yolact_package'])

from yolact_package.evaluate_colorfool import *
from yolact_package.eval import *
from yolact_package.data import *

# ---------------- Configuration ----------------
with open('./configuration.json', 'r') as f:
    configuration = json.load(f)

device = torch.device("cuda")
cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

cocoGt = COCO(f"{configuration['annotations']}/instances_val2017.json")

set_cfg("yolact_base_config")

dataset = COCODetection(f"{configuration['val2017']}/", f"{configuration['annotations']}/instances_val2017.json",
                        transform=NoneTransform(), has_gt=True)

conv_id_coco_dset = {dataset.ids[i]: i for i in range(len(dataset))}
prep_coco_cats()

yolact = Yolact()
yolact.to(device)
yolact.load_weights(configuration['weights_yolact'])
yolact.eval()

for param in yolact.parameters():
    param.requires_grad = False

args = parse_args('')

args.top_k = 5
cfg.mask_proto_debug = args.mask_proto_debug
yolact.detect.use_fast_nms = True
yolact.detect.use_cross_class_nms = False
cfg.train_masks = True
cfg.use_semantic_segmentation_loss = True
cfg.mask_proto_loss = 'l1'
cfg.pred_outs = False

conv_id_coco_dset = {dataset.ids[i]: i for i in range(len(dataset))}

# The id of images are formate with the id of the coco dataset
file_id = open(configuration['image_id_list'], 'rb')
imgs_to_attack_cocoid = pickle.load(file_id)
file_id.close()

imgs_to_attack = [conv_id_coco_dset[i] for i in imgs_to_attack_cocoid]


def generate_coco_file(dataset_indices, name, results, dataset, net, qf=-1):
    """This function generated the bbox and the mask files that follow the rules
    of COCOapi.

    Before using this function make sure that args.bbox_det_file and
    args.mask_det_file are well define.
    """
    if name not in results:
        results[name] = {}

    detections = Detections()
    list_naturalness = []

    for image_idx in tqdm(dataset_indices):

        _, _, _, h, w, _ = dataset.pull_item(image_idx)
        img_gt_open = Image.open(f'{configuration["result"]}/clean/{dataset.ids[image_idx]:012d}.png')
        img_gt = transforms.ToTensor()(img_gt_open).to(device)

        compression = '' if qf == -1 else f'_compress_{qf}'

        img_adv = Image.open(f'{configuration["result"]}/adversarials/adversarial_{dataset.ids[image_idx]:012d}{compression}.png')
        img_adv = transforms.ToTensor()(img_adv).to(device)

        denom = torch.sqrt(torch.mean(img_adv * img_adv))
        mse = torch.mean((img_adv - img_gt) ** 2)
        nrmse = torch.sqrt(mse) / denom
        psnr = - 10 * torch.log10(mse + 1e-10)
        ssim_score = ssim(img_adv.unsqueeze(0), img_gt.unsqueeze(0), data_range=1.)

        if qf is not -1:

            img_gt_open.save(f'{configuration["result"]}/clean/temp_comp.png', format='JPEG', quality=qf)
            img_gt_c = Image.open(f'{configuration["result"]}/clean/temp_comp.png')
            img_gt_c = transforms.ToTensor()(img_gt_c).to(device)

            mse_c = torch.mean((img_adv - img_gt_c) ** 2)
            nrmse_c = torch.sqrt(mse_c) / denom
            psnr_c = - 10 * torch.log10(mse_c + 1e-10)
            ssim_score_c = ssim(img_adv.unsqueeze(0), img_gt_c.unsqueeze(0), data_range=1.)

            list_naturalness.append((ssim_score.item(), mse.item(), nrmse.item(), psnr.item(),
                                     ssim_score_c.item(), mse_c.item(), nrmse_c.item(), psnr_c.item()))

        else:
            list_naturalness.append((ssim_score.item(), mse.item(), nrmse.item(), psnr.item()))

        img_adv = normalize_imagenet(img_adv)
        preds = net(img_adv.unsqueeze(0))

        my_prep_metrics(preds, h, w, dataset.ids[image_idx], detections)

    detections.dump()

    lenght = len(list_naturalness)
    results[name].update({"ssim": sum([i[0] for i in list_naturalness]) / lenght,
                          "mse": sum([i[1] for i in list_naturalness]) / lenght,
                          "nrmse": sum([i[2] for i in list_naturalness]) / lenght,
                          "psnr": sum([i[3] for i in list_naturalness]) / lenght})

    if qf is not -1:
        results[name].update({"ssim_comp": sum([i[4] for i in list_naturalness]) / lenght,
                              "mse_comp": sum([i[5] for i in list_naturalness]) / lenght,
                              "nrmse_comp": sum([i[6] for i in list_naturalness]) / lenght,
                              "psnr_comp": sum([i[7] for i in list_naturalness]) / lenght})


def evaluate_COCOapi(list_id_image, name, results, option, cocoGt):
    """
    With the coco API calculate the AP.
    """
    if name not in results:
        results[name] = {}

    for t in ['mask', 'bbox']:
        cocomask = cocoGt.loadRes(option[f'{t}_file_name'])
        cocoEval = COCOeval(cocoGt, cocomask, '{}'.format('segm' if t is 'mask' else t))
        cocoEval.params.imgIds = [dataset.ids[i] for i in list_id_image]
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        results[name].update({f'AP_{t}': cocoEval.stats[0], f'AP50_{t}': cocoEval.stats[1],
                              f'AP75_{t}': cocoEval.stats[2], f'AP_small_{t}': cocoEval.stats[3],
                              f'AP_medium_{t}': cocoEval.stats[4], f'AP_large_{t}': cocoEval.stats[5],
                              f'AR1_{t}': cocoEval.stats[6], f'AR10_{t}': cocoEval.stats[7],
                              f'AR100_{t}': cocoEval.stats[8], f'AR_small_{t}': cocoEval.stats[9],
                              f'AR_medium_{t}': cocoEval.stats[10], f'AR_large_{t}': cocoEval.stats[11]})


def print_result(results, end=', '):
    print("JPEG Quality Factor")
    for t in ['mask', 'bbox']:
        for m in ['AP_', 'AP50_']:
            print('{} : {}'.format(t, 'mAP  ' if m == 'AP_' else 'mAP50'), end=" = ")
            for i in ['All_compress_1', 'All_compress_10', 'All_compress_20', 'All_compress_30', 'All_compress_40', 'All_compress_50', 'All_compress_60', 'All_compress_70', 'All_compress_80', 'All_compress_90', 'All_compress_100', 'All']:
                print(f"{results[i][f'{m}{t}']*100:.5f}", end=end)
            print()
        print()

    for t in ['ssim', 'ssim_comp', 'psnr', 'psnr_comp', 'mse', 'mse_comp', 'nrmse', 'nrmse_comp']:
        print(f'{t}')
        for i in ['All_compress_1', 'All_compress_10', 'All_compress_20', 'All_compress_30', 'All_compress_40',
                  'All_compress_50', 'All_compress_60', 'All_compress_70', 'All_compress_80', 'All_compress_90',
                  'All_compress_100']:
            print(f"{results[i][t]:.7f}", end=end)
        if t[-5:] == '_comp':
            print(f"{results['All'][t[:-5]]:.7f}", end=end)
        else:
            print(f"{results['All'][t]:.7f}", end=end)
        print()
    print()


def generate_adversarial_image(dataset_list):
    for content_index_image in tqdm(dataset_list):
        clean_image_save = get_item(content_index_image, dataset, normalize=False)[0]
        save_image(clean_image_save, f'{configuration["result"]}/clean/{dataset.ids[content_index_image]:012d}.png')

        clean_image = get_item(content_index_image, dataset, normalize=False)[0].to(device)

        image_mask = Image.open(f'{configuration["mask_upper_shirt"]}/mask_{dataset.ids[content_index_image]:012d}.png')
        mask = transforms.ToTensor()(image_mask).to(device)

        image_patch = Image.open(f'{configuration["result"]}/patch_{dataset.ids[content_index_image]:012d}.png')
        patch = transforms.ToTensor()(image_patch).to(device)

        adversarial_img = torch.mul((1 - mask), clean_image) + torch.mul(mask, patch)
        save_image(adversarial_img, f'{configuration["result"]}/adversarials/adversarial_{dataset.ids[content_index_image]:012d}.png')


def generate_compress_adversarial_image(dataset_list, quality_factor):
    """
    Generate the compress images
    """
    for content_index_image in dataset_list:
        adversarial_img = Image.open(f'{configuration["result"]}/adversarials/adversarial_{dataset.ids[content_index_image]:012d}.png')

        compress_path = f'{configuration["result"]}/adversarials/adversarial_{dataset.ids[content_index_image]:012d}_compress_{quality_factor}.png'
        adversarial_img.save(compress_path, format='JPEG', quality=quality_factor)


def supress_compress_adversarial_image(dataset_list, quality_factor):
    for content_index_image in dataset_list:
        compress_path = f'{configuration["result"]}/adversarials/adversarial_{dataset.ids[content_index_image]:012d}_compress_{quality_factor}.png'
        os.remove(compress_path)


def my_evaluation(dataset_indices, option, dset=dataset, net=yolact, cocoGt=cocoGt):
    """
    This function perfome the Evaluation with
    """
    results = dict()
    generate_coco_file(dataset_indices, 'All', results, dset, net)

    # COCO Evaluation on uncompress images
    evaluate_COCOapi(dataset_indices, 'All', results, option, cocoGt)

    # JPEG Evaluation
    for quality_factor in tqdm([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        generate_compress_adversarial_image(dataset_indices, quality_factor)

        option['bbox_file_name'] = f'{configuration["result"]}/evaluation/bbox_detections_compress_{quality_factor}.json'
        option['mask_file_name'] = f'{configuration["result"]}/evaluation/mask_detections_compress_{quality_factor}.json'
        args.bbox_det_file = option['bbox_file_name']
        args.mask_det_file = option['mask_file_name']

        # Compress images
        generate_coco_file(dataset_indices, f'All_compress_{quality_factor}', results, dset, net, quality_factor)
        # Evaluate compress images
        evaluate_COCOapi(dataset_indices, f'All_compress_{quality_factor}', results, option, cocoGt)
        # Suppress compress images
        supress_compress_adversarial_image(dataset_indices, quality_factor)

    return results


if __name__ == '__main__':

    try:
        os.mkdir(f'{configuration["result"]}/evaluation')
        os.mkdir(f'{configuration["result"]}/adversarials')
        os.mkdir(f'{configuration["result"]}/clean')

        generate_adversarial_image(imgs_to_attack)

        args.bbox_det_file = f'{configuration["result"]}/evaluation/bbox_detections.json'
        args.mask_det_file = f'{configuration["result"]}/evaluation/mask_detections.json'
        op = {'bbox_file_name': args.bbox_det_file, 'mask_file_name': args.mask_det_file}

        results = my_evaluation(imgs_to_attack, op)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(f'{configuration["result"]}/evaluation/results.txt', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print_result(results, '\n')
            sys.stdout = original_stdout

        file_res = open(f'{configuration["result"]}/evaluation/results.pkl', 'wb')
        pickle.dump(results, file_res)
        file_res.close()

        shutil.rmtree(f'{configuration["result"]}/clean', ignore_errors=True)

    except FileExistsError:
        print('already done')
