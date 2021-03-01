"""File for new testing the model. TODO.

A more detailed explanation.
"""

import numpy as np
import torch
from transformers import BertModel
from lib import segmentation
from model import Model
import torchvision.transforms.transforms as T
from transformers import BertTokenizer
import PIL

import utils

import time


def main(args):
    tic = time.time()

    # Segmentation model.
    seg_model = segmentation.deeplabv3_resnet101(num_classes=2,
                                                 aux_loss=False,
                                                 pretrained=False,
                                                 args=args)

    print("hey from here: ", time.time() - tic)
    # BERT model.
    bert_model = BertModel.from_pretrained(args.ck_bert)



    # Load checkpoint.
    device = torch.device(args.device)
    ticAux = time.time()
    checkpoint = torch.load(args.resume, map_location=device)
    print("extra time", time.time() - ticAux)

    bert_model.load_state_dict(checkpoint["bert_model"], strict=False)
    seg_model.load_state_dict(checkpoint["model"], strict=False)

    # Define model and sent to device.
    model = Model(seg_model, bert_model)

    model.to(device)

    model.eval()

    print("loading of model time: ", time.time() - tic)
    tic = time.time()

    img_raw = PIL.Image.open(args.img)


    max_tokens = 20
    attention_mask = [0] * max_tokens
    padded_input_ids = [0] * max_tokens

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    input_ids = tokenizer.encode(text=args.sent,
                                      add_special_tokens=True)

    # truncation of tokens
    input_ids = input_ids[:max_tokens]

    padded_input_ids[:len(input_ids)] = input_ids
    attention_mask[:len(input_ids)] = [1]*len(input_ids)

    sents = torch.tensor(padded_input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    img = transforms(img_raw)
    imgs = img.unsqueeze(0)

    imgs, attentions, sents = \
        imgs.to(device), attention_mask.to(device), sents.to(device)

    print("prepare inputs: ", time.time() - tic)
    tic = time.time()


    with torch.no_grad():
        outputs = model(sents, attentions, imgs)
        masks = outputs.argmax(1)

    mask = masks.squeeze(0).cpu()

    print("forward model with no_grad: ", time.time() - tic)
    tic = time.time()


    utils.save_figure(img_raw, args.sent, mask, args.output)

    print("savefigure: ", time.time() - tic)
    tic = time.time()




if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    main(parser.parse_args())
