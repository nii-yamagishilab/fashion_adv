import torch
import random

def adversarial_noise(x, model, crit, target_y=None, nb_iterations=50, threshold=0.1, learning_rate=1, nb_class=80):
    """
    Args:
        X: A element from the dataloader, e.g. [images, (targets, masks, num_crowds)]
        model: The model we want to fool (here YOLACT)
        crit: The criterion for evaluate the model
        target_y: if target_y is None then the attack is untarget
        nb_iterations: maximal number of iteration
    Returns:
    """
    images, (targets, masks, num_crowds) = x

    x_tensor = images[0].unsqueeze(0).cuda()

    x_fooling = x_tensor.clone()
    x_fooling.requires_grad = True

    model.eval()
    temp = model(x_fooling)[0]['detection']

    indice = temp['score'] > threshold
    attacking_set = set(temp['class'][el].item() for el in range(len(temp['class'])) if indice[el])

    it = 0
    while it < nb_iterations:

        # Generate the target
        target = _chose_random_target(attacking_set, nb_class) if target_y is None else target_y

        for el in targets[0]: el[-1] = target

        # Adversarial
        model.train()
        prediction = model(x_fooling)

        pred_cpu = {}

        for n in prediction.items():
            pred_cpu[n[0]] = n[1].cpu()

        res = crit(model, pred_cpu, targets, masks, num_crowds)

        res['C'].backward()
        grad = x_fooling.grad

        x_fooling.data = x_fooling.data - learning_rate * grad / torch.norm(grad)

        # Evaluation
        model.eval()
        temp = model(x_fooling)[0]['detection']

        indice = temp['score'] > threshold
        temp_set = set(temp['class'][el].item() for el in range(len(temp['class'])) if indice[el])

        if target_y is None:
            if temp_set - attacking_set == temp_set:
                break
        else:
            if len(temp_set) == 1 and target_y in temp_set:
                break
        

        it += 1

    return x_fooling


def _chose_random_target(attacking_set, nb_class):
    target = random.randint(0, nb_class - 1)
    while target in attacking_set:
        target = random.randint(0, nb_class - 1)
    return target
