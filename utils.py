"""TODO

TODO
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def display_function(sent_ids, dataset, masks, directory):
    """Docs."""

    for sent_id, mask in zip(sent_ids, masks):
        sent = dataset.get_sent_raw(sent_id)
        image = dataset.get_image(sent_id)
        mask = mask[:image.size[1], :image.size[0]]  # Remove padding.

        plt.figure()
        plt.axis("off")
        plt.imshow(image)
        plt.text(0, 0, sent, fontsize=12)

        # mask definition
        img = np.ones((image.size[1], image.size[0], 3))
        color_mask = np.array([0, 255, 0]) / 255.0
        for i in range(3):
            img[:, :, i] = color_mask[i]
            plt.imshow(np.dstack((img, mask * 0.5)))

        figname = directory + str(sent_id) + ".png"
        plt.savefig(figname)
        plt.close()


def compute_jaccard_indices(masks, targets):
    """Docs."""

    cum_intersection, cum_union = 0, 0
    jaccard_indices = []

    for (mask, target) in zip(masks, targets):
        intersection = np.sum(np.logical_and(mask, target))
        union = np.sum(np.logical_or(mask, target))
        cum_intersection += intersection
        cum_union += union
        jaccard_indices.append(intersection/union)

    return jaccard_indices, cum_intersection, cum_union


def cat_list(images, fill_value=0):
    """Docs."""
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    """Docs."""
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def collate_fn_emb_berts(batch):
    """Docs."""
    images, targets, sents, attentions, sent_ids = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    batched_attentions = cat_list(attentions, fill_value=0)
    sents = torch.stack(sents)
    return batched_imgs, batched_targets, sents, batched_attentions, sent_ids
