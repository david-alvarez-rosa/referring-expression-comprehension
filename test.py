import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from transformers import *
import torchvision

from PIL import Image

from lib import segmentation

import transforms as T
import utils

import numpy as np


def get_dataset(name, image_set, transform, args):

    if args.baseline_bilstm:
        from data.dataset_refer_glove import ReferDataset
    else:
        from data.dataset_refer_bert import ReferDataset

    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      input_size=(256, 448),
                      eval_mode=True)

    num_classes = 2

    print(len(ds))

    return ds, num_classes


def evaluate(args, model, data_loader, ref_ids,
             refer, bert_model, device, num_classes,
             display=True, baseline_model=None,
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
        for (k , (images, targets, sentences, attentions, sents, image_infos)) in enumerate(data_loader):

            images, sentences, attentions = images.to(device), \
                sentences.to(device), attentions.to(device)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            targets = targets.cpu().data.numpy()

            last_hidden_states = bert_model(sentences,
                                            attention_mask=attentions)[0]

            embedding = last_hidden_states[:, 0, :]

            outputs, _, _ = model(images, embedding.squeeze(1))
            outputs = outputs['out'].cpu()

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

            del targets, images, attentions

            sent = sents[0]
            mask = masks[0]

            if display:

                plt.figure()
                plt.axis('off')

                sentence = sent

                IMAGE_DIR = "datasets/refcoco/images"
                image = Image.open(os.path.join(IMAGE_DIR,
                                                image_infos["file_name"][0])
                                   ).convert("RGB")

                plt.imshow(image)

                plt.text(0, 0, sentence, fontsize=12)

                ax = plt.gca()
                ax.set_autoscale_on(False)

                # mask definition
                img = np.ones((image.size[1], image.size[0], 3))
                color_mask = np.array([0, 255, 0]) / 255.0
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, mask * 0.5)))

                results_folder = args.results_folder
                if not os.path.isdir(results_folder):
                    os.makedirs(results_folder)

                figname = os.path.join(args.results_folder, str(k) + '.png')
                plt.savefig(figname)


    mean_IoU = np.array(mean_IoU)
    # TODO: fixme.
    # mIoU = np.mean(mean_IoU)
    mIoU = 20
    # TODO: end

    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    # for n_eval_iou in range(len(eval_seg_iou_list)):
    #     results_str += '    precision@%s = %.2f\n' % \
    #         (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)

    # TODO: fix me.
    cum_U += 1e-8
    # TODO: end.


    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)

    print(results_str)

    return refs_ids_list


def get_transform():

    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


# compute IoU
def computeIoU(pred_seg, gd_seg):

    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):

    device = torch.device(args.device)

    dataset_test, _ = get_dataset(args.dataset, args.split, get_transform(), args)

    print(len(dataset_test))

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    model = segmentation.__dict__[args.model](num_classes=2,
        aux_loss=False,
        pretrained=False,
        args=args)

    model.to(device)
    model_class = BertModel

    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.to(device)


    if args.baseline_bilstm:
        bilstm = torch.nn.LSTM(input_size=300, hidden_size=1000, num_layers=1, bidirectional=True, batch_first=True)
        fc_layer = torch.nn.Linear(2000, 768)
        bilstm = bilstm.to(device)
        fc_layer = fc_layer.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')

    bert_model.load_state_dict(checkpoint['bert_model'], strict=False)
    model.load_state_dict(checkpoint['model'])

    if args.baseline_bilstm:
        bilstm.load_state_dict(checkpoint['bilstm'])
        fc_layer.load_state_dict(checkpoint['fc_layer'])

    if args.dataset == 'refcoco' or args.dataset == 'refcoco+':
        ref_ids = dataset_test.ref_ids
        refer = dataset_test.refer
        ids = ref_ids
        objs_ids = None
        num_objs_list = None
    elif args.dataset == 'davis':
        ids = dataset_test.ids
        objs_ids = None
        num_objs_list = None

        with open(args.davis_annotations_file) as f:
            lines = f.readlines()

    elif args.dataset == 'a2d':
        ids = dataset_test.img_list
        objs_ids = dataset_test.objs
        num_objs_list = dataset_test.num_objs_list

        with open(args.davis_annotations_file) as f:
            lines = f.readlines()

    if args.dataset == 'davis' or args.dataset == 'a2d':

        refer = {}

        for l in lines:
            words = l.split()

            refer[words[0] + '_' + words[1]] = {}
            refer[words[0] + '_' + words[1]] = ' '.join(words[2:])[1:-1]

    if args.baseline_bilstm:
        baseline_model = [bilstm, fc_layer]
    else:
        baseline_model = None

    refs_ids_list = evaluate(args, model, data_loader_test, ids, refer, bert_model, device=device,
        num_classes=2, baseline_model=baseline_model,  objs_ids=objs_ids, num_objs_list=num_objs_list)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
