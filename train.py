"""TODO"""

import torch
from functools import reduce
import operator
from transformers import BertModel
from lib import segmentation
import transforms as T
import utils
import gc
from dataset import ReferDataset




import time




from aux import Model




def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30
    epochs"""
    lr = args.lr - args.lr_specific_decrease*epoch
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_transform(train, base_size=520, crop_size=480):
    """TODO"""
    min_size = int((0.8 if train else 1.0) * base_size)
    max_size = int((0.8 if train else 1.0) * base_size)

    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))

    if train:
        transforms.append(T.RandomCrop(crop_size))

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def criterion(maks, targets):
    """TODO"""
    losses = {}
    for name, x in inputs.items():
        losses[name] = torch.nn.functional.cross_entropy(x, targets, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def train_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args):
    model.train()
    loss_value = 0
    num_its = 0

    for imgs, targets, sents, attentions, sent_ids in data_loader:
        tic = time.time()

        num_its += 1

        imgs, attentions, sents, targets = \
            imgs.to(device), attentions.to(device), \
            sents.to(device), targets.to(device)

        sents = sents.squeeze(1)
        attentions = attentions.squeeze(1)

        outputs = model(sents, attentions, imgs)
        # masks = outputs.argmax(1)

        loss = torch.nn.functional.cross_entropy(outputs, targets, ignore_index=255)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.linear_lr:
            adjust_learning_rate(optimizer, epoch, args)
        else:
            lr_scheduler.step()

        loss_value += loss.item()

        del imgs, targets, sents, attentions, loss, outputs

        gc.collect()
        torch.cuda.empty_cache()

        print(loss_value/num_its)
        print(time.time() - tic)

    print("="*79)
    print(loss_value/num_its)


def main(args):
    device = torch.device(args.device)

    # Train dataset.
    dataset = ReferDataset(args, transforms=get_transform(train=True))
    train_sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        #collate_fn=utils.collate_fn_emb_berts,
        drop_last=True)

    # Validation dataset.
    dataset_val = ReferDataset(args, transforms=get_transform(train=False))
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_val, batch_size=1,
        sampler=val_sampler, num_workers=args.workers,
        #collate_fn=utils.collate_fn_emb_berts
    )

    # Segmentation model.
    seg_model = segmentation.__dict__[args.seg_model](num_classes=2,
                                              aux_loss=args.aux_loss,
                                              pretrained=args.pretrained,
                                              args=args)

    # BERT model.
    bert_model = BertModel.from_pretrained(args.ck_bert)

    if args.pretrained_refvos:
        checkpoint = torch.load(args.ck_pretrained_refvos)
        seg_model.load_state_dict(checkpoint["seg_model"])
        bert_model.load_state_dict(checkpoint["bert_model"])
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        seg_model.load_state_dict(checkpoint["seg_model"])

    params_to_optimize = [
        {"params": [p for p in seg_model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in seg_model.classifier.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat, [[p for p in bert_model.encoder.layer[i].parameters() if p.requires_grad] for i in range(10)])},
        {"params": [p for p in bert_model.pooler.parameters() if p.requires_grad]}
    ]

    model = Model(seg_model, bert_model)
    model.to(device)

    if args.aux_loss:
        params = [p for p in seg_model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.fixed_lr:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: args.lr_specific)
    elif args.linear_lr:
        lr_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    t_iou = 0

    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])

        if not args.fixed_lr:
            if not args.linear_lr:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    for epoch in range(args.epochs):
        train_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args)

        # only save if checkpoint improves
        if False and t_iou < iou: # TODO: recompute IoU.
            print("Better epoch: {}\n".format(epoch))

            dict_to_save = {"seg_model": seg_model.state_dict(),
                            "bert_model": bert_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args}

            if not args.linear_lr:
                dict_to_save["lr_scheduler"] = lr_scheduler.state_dict()

            utils.save_on_master(dict_to_save, args.output_dir + "model_best_{}.pth".format(args.model_id))

            t_iou = iou

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    main(parser.parse_args())
