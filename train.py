"""Docs."""

import datetime
import os
import time
import torch
from functools import reduce
import operator
from transformers import BertModel
from lib import segmentation
import transforms as T
import utils
import numpy as np
import gc
from dataset import ReferDataset


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr - args.lr_specific_decrease*epoch
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# IoU calculation for proper validation
def IoU(pred, gt):
    """Docs."""
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection)/float(union)

    return iou


def get_transform(train, base_size=520, crop_size=480):
    """Docs."""
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


def criterion(inputs, targets, args):
    """Docs."""
    losses = {}
    for name, x in inputs.items():
        losses[name] = torch.nn.functional.cross_entropy(x, targets, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(args, model, dataset, data_loader,
             refer, bert_model, device, display=True, baseline_model=None,
             objs_ids=None, num_objs_list=None):

    model.eval()
    refs_ids_list = []

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for imgs, targets, sents, attentions, sent_ids in data_loader:

            imgs, sents, attentions = imgs.to(device), \
                sents.to(device), attentions.to(device)

            sents = sents.squeeze(1)
            attentions = attentions.squeeze(1)

            targets = targets.data.numpy()

            last_hidden_states = bert_model(sents,
                                            attention_mask=attentions)[0]

            embedding = last_hidden_states[:, 0, :]

            outputs, _, _ = model(imgs, embedding.squeeze(1))
            outputs = outputs["out"]

            masks = outputs.argmax(1).data.numpy()

            I, U = computeIoU(masks, targets)

            if U == 0:
                this_iou = 0.0
            else:
                this_iou = I*1.0/U

            mean_IoU.append(this_iou)

            cum_I += I
            cum_U += U

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del targets, imgs, attentions

            sent_id = int(sent_ids[0])
            sent = dataset.get_sent_raw(sent_id)
            mask = masks[0]

            if display:
                sentence = sent

                imgs = dataset.get_image(sent_id)

                plt.figure()
                plt.axis("off")
                plt.imshow(imgs)
                plt.text(0, 0, sentence, fontsize=12)

                # mask definition
                img = np.ones((imgs.size[1], imgs.size[0], 3))
                color_mask = np.array([0, 255, 0]) / 255.0
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                    plt.imshow(np.dstack((img, mask * 0.5)))

                results_folder = args.results_folder
                if not os.path.isdir(results_folder):
                    os.makedirs(results_folder)

                figname = os.path.join(args.results_folder, str(sent_id) + ".png")
                plt.savefig(figname)
                plt.close()


    mean_IoU = np.array(mean_IoU)
    # TODO: fixme.
    # mIoU = np.mean(mean_IoU)
    mIoU = 20
    # TODO: end

    print("Final results:")
    print("Mean IoU is %.2f\n" % (mIoU*100.))
    results_str = ""
    # for n_eval_iou in range(len(eval_seg_iou_list)):
    #     results_str += "    precision@%s = %.2f\n" % \
        #         (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)

    # TODO: fix me.
    cum_U += 1e-8
    # TODO: end.

    results_str += "    overall IoU = %.2f\n" % (cum_I * 100. / cum_U)

    print(results_str)

    return refs_ids_list


class allModel():
    def __init__(model, bert_model):
        self.model = model
        self.bert_model = bert_model

    def forward():
        last_hidden_states = self.bert_model(sents, attention_mask=attentions)[0]
        embedding = last_hidden_states[:, 0, :]
        output, vis_emb, lan_emb = model(imgs, embedding.squeeze(1))

        return output


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args, print_freq, iterations, bert_model, baseline_model):
    model.train()
    loss = 0
    num_its = 0

    for imgs, targets, sents, attentions, sent_ids in data_loader:
        num_its += 1

        imgs, targets, sents, attentions = imgs.to(device), targets.to(device), sents.to(device), attentions.to(device)

        sents = sents.squeeze(1)
        attentions = attentions.squeeze(1)

        last_hidden_states = bert_model(sents, attention_mask=attentions)[0]

        embedding = last_hidden_states[:, 0, :]
        output, _, _ = model(imgs, embedding.squeeze(1))

        loss_class = criterion(output, targets, args)

        optimizer.zero_grad()
        loss_class.backward()
        optimizer.step()

        if args.linear_lr:
            adjust_learning_rate(optimizer, epoch, args)
        else:
            lr_scheduler.step()

        loss += loss_class.item()
        iterations += 1

        del imgs, targets, sents, attentions, loss_class, embedding, output, last_hidden_states

        gc.collect()
        torch.cuda.empty_cache()

        print(loss/num_its)


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

    # Model definition.
    model = segmentation.__dict__[args.model](num_classes=2,
                                              aux_loss=args.aux_loss,
                                              pretrained=args.pretrained,
                                              args=args)
    model_class = BertModel
    bert_model = model_class.from_pretrained(args.ck_bert)

    if args.pretrained_refvos:
        checkpoint = torch.load(args.ck_pretrained_refvos)
        model.load_state_dict(checkpoint["model"])
        bert_model.load_state_dict(checkpoint["bert_model"])
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    model = model.cuda()
    bert_model = bert_model.cuda()

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat, [[p for p in bert_model.encoder.layer[i].parameters() if p.requires_grad] for i in range(10)])},
        {"params": [p for p in bert_model.pooler.parameters() if p.requires_grad]}
    ]

    if args.aux_loss:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
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

    model_dir = os.path.join("./models/", args.model_id)

    start_time = time.time()

    iterations = 0
    t_iou = 0

    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])

        if not args.fixed_lr:
            if not args.linear_lr:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if args.baseline_bilstm:
        baseline_model = [bilstm, fc_layer]
    else:
        baseline_model = None

    for epoch in range(args.epochs):

        train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args, args.print_freq, iterations, bert_model, baseline_model=baseline_model)

        # refs_ids_list = evaluate(args, model, dataset_val, data_loader_test, refer, bert_model, device=device, num_classes=2, baseline_model=baseline_model,  objs_ids=objs_ids, num_objs_list=num_objs_list)

        # only save if checkpoint improves
        if False and t_iou < iou: # TODO: recompute IoU.
            print("Better epoch: {}\n".format(epoch))

            if args.baseline_bilstm:
                utils.save_on_master(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "bilstm": bilstm.state_dict(),
                        "fc_layer": fc_layer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    os.path.join(args.output_dir, "model_best_{}.pth".format(args.model_id)))

            else:
                dict_to_save = {"model": model.state_dict(),
                                "bert_model": bert_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "args": args}

                if not args.linear_lr:
                    dict_to_save["lr_scheduler"] = lr_scheduler.state_dict()

                utils.save_on_master(dict_to_save, os.path.join(args.output_dir, "model_best_{}.pth".format(args.model_id)))

            t_iou = iou

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()

    main(args)
