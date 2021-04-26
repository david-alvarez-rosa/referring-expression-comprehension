"""File for testing the model.

A more detailed explanation.
"""

import time
import numpy as np
import torch
from transformers import BertModel
import transforms
from lib import segmentation
from dataset import ReferDataset
import utils
from model import Model


def evaluate(data_loader, model, device, dataset=None, results_dir=None):
    """Evaluate the model in the given dataset."""

    model.eval()

    loss_value = 0
    cum_intersection, cum_union = 0, 0
    jaccard_indices = []

    tic = time.time()

    for imgs, targets, sents, attentions, sent_ids in data_loader:
        print(time.time() - tic)
        tic = time.time()

        imgs, attentions, sents, targets = \
            imgs.to(device), attentions.to(device), \
            sents.to(device), targets.to(device)

        sents = sents.squeeze(1)
        attentions = attentions.squeeze(1)

        with torch.no_grad():
            outputs = model(sents, attentions, imgs)
            loss = torch.nn.functional.cross_entropy(outputs, targets, ignore_index=255)
            masks = outputs.argmax(1)

        loss_value += loss.item()

        jaccard_indices_batch, intersection, union = \
            utils.compute_jaccard_indices(masks, targets)
        jaccard_indices += jaccard_indices_batch
        cum_intersection += intersection
        cum_union += union

        if results_dir is not None:
            utils.save_output(dataset, sent_ids, masks, results_dir)

        # Release memory.
        del imgs, targets, sents, attentions, sent_ids

        # Added.
        print("loss: {:.4f}".format(loss_value/len(data_loader)))
        print("len_jaccard_indices: ", len(jaccard_indices))
        mean_jaccard_index = np.mean(np.array(jaccard_indices))
        print("Mean IoU is {:.4f}.".format(mean_jaccard_index))
        print("Overall IoU is {:.4f}.".format(cum_intersection/cum_union))
        print("\n\n")

    print("\n"*10)
    print("loss: {:.4f}".format(loss_value/len(data_loader)))
    print("jaccard_indices: ", jaccard_indices)
    mean_jaccard_index = np.mean(np.array(jaccard_indices))
    print("Mean IoU is {:.4f}.".format(mean_jaccard_index))
    print("Overall IoU is {:.4f}.".format(cum_intersection/cum_union))


def main(args):
    # Define dataset.
    dataset = ReferDataset(args,
                           split=args.split,
                           transforms=transforms.get_transform())



    # Modification.
    # print(len(dataset))
    # dataset = torch.utils.data.Subset(dataset, torch.arange(1))



    data_loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.workers,
                                         collate_fn=utils.collate_fn_emb_berts)

    # Segmentation model.
    seg_model = segmentation.deeplabv3_resnet101(num_classes=2,
                                                 aux_loss=False,
                                                 pretrained=False,
                                                 args=args)

    # BERT model.
    bert_model = BertModel.from_pretrained(args.ck_bert)

    # Load checkpoint.
    checkpoint = torch.load(args.resume, map_location="cpu")
    bert_model.load_state_dict(checkpoint["bert_model"], strict=False)
    seg_model.load_state_dict(checkpoint["model"], strict=False)

    # Define model and sent to device.
    model = Model(seg_model, bert_model)
    device = torch.device(args.device)
    model.to(device)

    evaluate(data_loader=data_loader,
             model=model,
             device=device,
             dataset=dataset,
             results_dir=args.results_dir)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    main(parser.parse_args())
