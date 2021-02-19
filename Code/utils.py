"""TODO

TODO
"""
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

import errno
import os




def save_figure(img, sent, mask, file_name, color="#0066ff"):
    """TODO"""

    plt.figure()
    plt.axis("off")
    plt.imshow(img)
    plt.text(0, 0, sent, fontsize=12)


    color_rgb = np.array(matplotlib.colors.to_rgb(color))
    color_mask = color_rgb * np.ones((img.size[1], img.size[0], 3))
    plt.imshow(np.dstack((color_mask, mask * 0.5)))

    plt.savefig(file_name)
    plt.close()





def save_output(dataset, sent_ids, masks, directory):
    """TODO"""

    # Ensure directory exists.
    mkdir(directory)

    for sent_id, mask in zip(sent_ids, masks):
        sent = dataset.get_sent_raw(sent_id)
        image = dataset.get_image(sent_id)
        # Remove padding and sent to CPU.
        mask = mask[:image.size[1], :image.size[0]].cpu()

        plt.figure()
        plt.axis("off")
        plt.imshow(image)
        plt.text(0, 0, sent, fontsize=12)

        # Mask definition.
        img = np.ones((image.size[1], image.size[0], 3))
        color_mask = np.array([0, 255, 0]) / 255.0
        for i in range(3):
            img[:, :, i] = color_mask[i]
            plt.imshow(np.dstack((img, mask * 0.5)))

        figname = directory + str(sent_id) + ".png"
        plt.savefig(figname)
        plt.close()


def compute_jaccard_indices(masks, targets):
    """TODO"""

    cum_intersection, cum_union = 0, 0
    jaccard_indices = []

    for (mask, target) in zip(masks, targets):
        intersection = torch.sum(torch.logical_and(mask, target)).cpu().numpy()
        union = torch.sum(torch.logical_or(mask, target)).cpu().numpy()
        cum_intersection += intersection
        cum_union += union
        jaccard_indices.append(intersection/union)

    return jaccard_indices, cum_intersection, cum_union


def cat_list(imgs, fill_value=0):
    """TODO"""

    max_size = tuple(max(s) for s in zip(*[img.shape for img in imgs]))
    batch_shape = (len(imgs),) + max_size
    batched_imgs = imgs[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(imgs, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    """TODO"""

    imgs, targets = list(zip(*batch))
    batched_imgs = cat_list(imgs, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def collate_fn_emb_berts(batch):
    """TODO"""

    imgs, targets, sents, attentions, sent_ids = list(zip(*batch))
    batched_imgs = cat_list(imgs, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    batched_attentions = cat_list(attentions, fill_value=0)
    sents = torch.stack(sents)
    return batched_imgs, batched_targets, sents, batched_attentions, sent_ids



















def mkdir(path):
    """Creates a directory. equivalent to using mkdir -p on the command line"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
