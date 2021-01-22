"""TODO: this is the test.py file

A more detailed explanation.
"""
import numpy as np
import torch
from transformers import BertModel
import transforms as T
from lib import segmentation
from dataset import ReferDataset
import utils



from aux import Model




def evaluate(args, dataset, loader, model, device):
    """Docs."""
    model.eval()

    cum_intersection, cum_union = 0, 0
    jaccard_indices = []

    for imgs, targets, sents, attentions, sent_ids in loader:
        imgs, attentions, sents, targets = \
            imgs.to(device), attentions.to(device), \
            sents.to(device), targets.to(device)

        sents = sents.squeeze(1)
        attentions = attentions.squeeze(1)

        with torch.no_grad():
            outputs = model(sents, attentions, imgs)
            mask = outputs.argmax(1)

        jaccard_indices_batch, intersection, union = \
            utils.compute_jaccard_indices(masks, targets)
        print("jaccard_indices_batch: ", jaccard_indices_batch)

        jaccard_indices += jaccard_indices_batch
        cum_intersection += intersection
        cum_union += union

        del targets, imgs, attentions

        utils.save_output(dataset, sent_ids, masks, args.results_folder)

    mean_jaccard_index = np.mean(np.array(jaccard_indices))
    print("Final results:")
    print("Mean IoU is {:.4f}.".format(mean_jaccard_index))
    print("Overall IoU is {:.4f}.".format(cum_intersection/cum_union))


def get_transform():
    """TODO"""
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def main(args):
    device = torch.device(args.device)

    # Define dataset.
    dataset = ReferDataset(args, transforms=get_transform())
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.workers,
                                         collate_fn=utils.collate_fn_emb_berts)

    # Segmentation model.
    seg_model = segmentation.__dict__[args.seg_model](num_classes=2,
                                              aux_loss=False,
                                              pretrained=False,
                                              args=args)

    # BERT model.
    bert_model = BertModel.from_pretrained(args.ck_bert)

    # Load checkpoint.
    checkpoint = torch.load(args.resume, map_location="cpu")
    bert_model.load_state_dict(checkpoint["bert_model"], strict=False)
    seg_model.load_state_dict(checkpoint["model"], strict=False)

    model = Model(seg_model, bert_model)
    model.to(device)

    evaluate(args, dataset, loader, model, device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    main(parser.parse_args())
