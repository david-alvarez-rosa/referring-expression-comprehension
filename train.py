"""TODO"""

import time
import torch
from functools import reduce
import operator
from transformers import BertModel
from lib import segmentation
import transforms
import utils
import gc
from dataset import ReferDataset
from model import Model



import test


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30
    epochs"""
    lr = args.lr - args.lr_specific_decrease*epoch
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args):
    """Train and compute loss and accuracy on train dataset_train.
    """

    model.train()

    for imgs, targets, sents, attentions, sent_ids in data_loader:
        # Sent data to device.
        imgs, attentions, sents, targets = \
            imgs.to(device), attentions.to(device), \
            sents.to(device), targets.to(device)
        sents = sents.squeeze(1)
        attentions = attentions.squeeze(1)

        # Compute model output and loss.
        outputs = model(sents, attentions, imgs)
        loss = torch.nn.functional.cross_entropy(outputs, targets, ignore_index=255)

        # Backpropagate.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adjust learning rate.
        if args.linear_lr:
            adjust_learning_rate(optimizer, epoch, args)
        else:
            lr_scheduler.step()

        # Release memory.
        del imgs, targets, sents, attentions, loss, outputs
        gc.collect()
        torch.cuda.empty_cache()


def new_epoch(model, optimizer, dataset_train, dataset_val, data_loader_train, data_loader_val, lr_scheduler, device, epoch, args):
    time_start_epoch = time.time()

    # Train.
    time_start_train = time.time()
    train_epoch(model=model,
                optimizer=optimizer,
                data_loader=data_loader_train,
                lr_scheduler=lr_scheduler,
                device=device,
                epoch=epoch,
                args=args)
    time_end_train = time.time()

    # # Evaluate in train dataset.
    # print("--- Train ---")
    # time_start_evaluate_train = time.time()
    # test.evaluate(data_loader=data_loader_train,
    #               model=model,
    #               device=device,
    #               dataset=dataset_train,
    #               results_dir=args.results_dir + str(epoch + 1) + "/")
    # time_end_evaluate_train = time.time()

    # Evaluate in validation dataset.
    print("\n--- Validation ---")
    time_start_evaluate_val = time.time()
    test.evaluate(data_loader=data_loader_val,
                  model=model,
                  device=device,
                  dataset=dataset_train,
                  results_dir=args.results_dir + str(epoch + 1) + "/")
    time_end_evaluate_val = time.time()

    # Times.
    time_end_epoch = time.time()

    print("\n--- Time ---")
    print("time_train: {:.2f}s".format(time_end_train - time_start_train))
    # print("time_evaluate_train: {:.2f}s".format(time_end_evaluate_train - time_start_evaluate_train))
    print("time_evaluate_test: {:.2f}s".
          format(time_end_evaluate_val - time_start_evaluate_val))
    print("time_epoch: {:.2f}s".format(time_end_epoch - time_start_epoch))


def main(args):
    device = torch.device(args.device)

    # Train dataset.
    dataset_train = ReferDataset(args,
                                 transforms=transforms.get_transform(train=True))
    data_sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        sampler=data_sampler_train,
        num_workers=args.workers,
        #collate_fn=utils.collate_fn_emb_berts,
        drop_last=True)

    # Validation dataset.
    dataset_val = ReferDataset(args,
                               transforms=transforms.get_transform())
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=1,
        sampler=sampler_val,
        num_workers=args.workers,
        #collate_fn=utils.collate_fn_emb_berts
    )

    # Segmentation model.
    seg_model = segmentation.deeplabv3_resnet101(num_classes=2,
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
            lambda x: (1 - x / (len(data_loader_train) * args.epochs)) ** 0.9)

    t_iou = 0

    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])

        if not args.fixed_lr:
            if not args.linear_lr:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    for epoch in range(args.epochs):
        print(("="*25 + " Epoch {}/{} " + "="*25).format(epoch + 1, args.epochs))
        new_epoch(model=model,
                  optimizer=optimizer,
                  dataset_train=dataset_train,
                  dataset_val=dataset_val,
                  data_loader_train=data_loader_train,
                  data_loader_val=data_loader_val,
                  lr_scheduler=lr_scheduler,
                  device=device,
                  epoch=epoch,
                  args=args)

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
