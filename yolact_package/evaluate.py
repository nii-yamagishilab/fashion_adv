from torchvision import transforms
from PIL import Image

from layers.output_utils import postprocess
from eval import Detections


def my_prep_metrics(dets, h, w, image_id, detections, args):
    """
    """

    classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    if classes.size(0) == 0:
        return

    classes = list(classes.cpu().numpy().astype(int))
    if isinstance(scores, list):
        box_scores = list(scores[0].cpu().numpy().astype(float))
        mask_scores = list(scores[1].cpu().numpy().astype(float))
    else:
        scores = list(scores.cpu().data.numpy().astype(float))
        box_scores = scores
        mask_scores = scores

    masks = masks.view(-1, h * w).cuda()
    boxes = boxes.cuda()

    boxes = boxes.cpu().detach().numpy()
    masks = masks.view(-1, h, w).cpu().detach().numpy()

    for i in range(masks.shape[0]):
        # Make sure that the bounding box actually makes sense and a mask was produced
        if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0 and classes[i] == 0:
            detections.add_bbox(image_id, classes[i], boxes[i, :], box_scores[i])
            detections.add_mask(image_id, classes[i], masks[i, :, :], mask_scores[i])
    return


def generate_coco_file(dataset_indices: list, path, dataset, net, args):
    """This function generated the bbox and the mask files that follow the rules
    of COCOapi.

    Before using this function make sure that args.bbox_det_file and
    args.mask_det_file are well define, otherwise this function will

    """
    args.bbox_det_file = f'{path}/bbox_detections.json'
    args.mask_det_file = f'{path}/mask_detections.json'

    detections = Detections()

    for image_idx in dataset_indices:
        _, _, _, h, w, _ = dataset.pull_item(image_idx)

        img_adv = Image.open(f'{path}/adversarial_{dataset.ids[image_idx]:012d}.png')
        img_adv = transforms.ToTensor()(img_adv).cuda()

        img_adv = _normalize_imagenet(img_adv)

        preds = net(img_adv.unsqueeze(0))

        my_prep_metrics(preds, h, w, dataset.ids[image_idx], detections, args)

    detections.dump()


_MEAN_imagenet = [0.485, 0.456, 0.406]
_STD_imagenet = [0.229, 0.224, 0.225]
_normalize_imagenet = transforms.Compose([transforms.Normalize(mean=_MEAN_imagenet, std=_STD_imagenet)])
