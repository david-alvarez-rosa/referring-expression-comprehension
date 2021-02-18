"""File for new testing the model. TODO.

A more detailed explanation.
"""

import numpy as np
import torch
from transformers import BertModel
from lib import segmentation
from dataset import ReferDataset
import utils
from model import Model


import torchvision.transforms.transforms as T
from transformers import BertTokenizer
from PIL import Image


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

    file_name = args.file_name
    sent_raw = args.sent

    img = Image.open(file_name)
    imageOriginal = img



    max_tokens = 20
    attention_mask = [0] * max_tokens
    padded_input_ids = [0] * max_tokens

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    input_ids = tokenizer.encode(text=sent_raw,
                                      add_special_tokens=True)

    # truncation of tokens
    input_ids = input_ids[:max_tokens]

    padded_input_ids[:len(input_ids)] = input_ids
    attention_mask[:len(input_ids)] = [1]*len(input_ids)

    tensor_embeddings = torch.tensor(padded_input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)




    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    img = transforms(img)

    imgs = img.to(device)
    attentions = attention_mask.to(device)
    sents = tensor_embeddings.to(device)


    imgs = imgs.unsqueeze(0)

    print(sents.shape)

    with torch.no_grad():
        outputs = model(sents, attentions, imgs)
        masks = outputs.argmax(1)


    print(masks)
    print(masks.shape)
    print("here i should savethe output, but now sent_id does not exist")


    mask = masks.squeeze(0).cpu()
    # mask = mask[:image.size[1], :image.size[0]].cpu()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.axis("off")
    plt.imshow(imageOriginal)
    plt.text(0, 0, sent_raw, fontsize=12)

    # Mask definition.
    img = np.ones((imageOriginal.size[1], imageOriginal.size[0], 3))
    color_mask = np.array([0, 255, 0]) / 255.0
    for i in range(3):
        img[:, :, i] = color_mask[i]
        plt.imshow(np.dstack((img, mask * 0.5)))

    figname = "hola.png"
    plt.savefig(figname)
    plt.close()




if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    main(parser.parse_args())
