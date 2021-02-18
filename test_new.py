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


def main(args):
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








    model.eval()


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

    with torch.no_grad():
        outputs = model(sents, attentions, imgs)
        masks = outputs.argmax(1)

    mask = masks.squeeze(0).cpu()

    utils.save_figure(img_raw, args.sent, mask, args.output_file)





if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    main(parser.parse_args())
