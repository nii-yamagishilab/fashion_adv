from torchvision import transforms
import pickle


MEAN_imagenet = [0.485, 0.456, 0.406]
MINUS_MEAN = [-i for i in MEAN_imagenet]
STD_imagenet = [0.229, 0.224, 0.225]
MINUS_STD = [-i for i in STD_imagenet]
STD_INV = [1 / i for i in STD_imagenet]

normalize_imagenet = transforms.Compose([transforms.Normalize(mean=MEAN_imagenet, std=STD_imagenet)])
normalize_mean = transforms.Compose([transforms.Normalize(mean=MEAN_imagenet, std=[1, 1, 1])])
normalize_std = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=STD_imagenet)])

undo_normalize_imagenet = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=STD_INV),
                                              transforms.Normalize(mean=MINUS_MEAN, std=[1, 1, 1])])
undo_mean_imagenet = transforms.Compose([transforms.Normalize(mean=MINUS_MEAN, std=[1, 1, 1])])
undo_std_imagenet = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=STD_INV)])


def apply_compose(img, compose):
    return compose(img.squeeze(0)).unsqueeze(0)


def normalize(img):
    return normalize_imagenet(img.squeeze(0)).unsqueeze(0)


def undo_normalize(img):
    return undo_normalize_imagenet(img.squeeze(0)).unsqueeze(0)


colors_palette = ["#000012", "#0f0c31", "#04595d", "#53849c", "#98b0dd", "#d79e1c",
                  "#f6875c", "#fb5318", "#ee040e", "#bc0b28", "#88013e", "#500080"]


def get_item(indice, data, normalize=True):
    image, (targets, masks, num_crowds) = data[indice]
    if not normalize:
        return image, targets, masks, num_crowds
    return normalize_imagenet(image).unsqueeze(0), targets, masks, num_crowds

def get_image_to_attack(dataset, path):
    conv_id_coco_dset = {dataset.ids[i]: i for i in range(len(dataset))}

    with open(path, 'rb') as file_id:
        imgs_to_attack_cocoid = pickle.load(file_id)

    return [conv_id_coco_dset[i] for i in imgs_to_attack_cocoid]