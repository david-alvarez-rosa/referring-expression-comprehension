"""Explanation.

TODO: explanation.

More text.
"""

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from transformers import BertTokenizer
from refer import Refer


class ReferDataset(data.Dataset):
    """Class for the Refer dataset.


    """

    def __init__(self, args, split, transforms, dataset_root="../Dataset/refcoco/",
                 ref_file="refs(unc).p", ann_file="annotations.json"):
        """Creates class.

        More comments.
        """

        self.transforms = transforms
        self.dataset_root = dataset_root
        self.image_root = args.image_root
        self.refer = Refer(dataset_root + ann_file, dataset_root + ref_file)

        self.max_tokens = 20

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.sent_ids = self.refer.get_sent_ids(split=split)

    def __len__(self):
        """Size of dataset."""
        return len(self.sent_ids)

    def __getitem__(self, index):
        sent_id = self.sent_ids[index]
        sent = self.refer.sents[sent_id]["raw"]

        ref_id = self.refer.sent_to_ref[sent_id]
        ann_id = self.refer.ref_to_ann[ref_id]
        ann = self.refer.anns[ann_id]

        # Get image.
        img_id = ann["image_id"]
        img_name = self.refer.imgs[img_id]["file_name"]
        img = Image.open(self.image_root + img_name)

        mask = self.refer.ann_to_mask(ann)
        mask = Image.fromarray(mask.astype(np.uint8), mode="P")

        img, mask = self.transforms(img, mask)

        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens

        input_ids = self.tokenizer.encode(text=sent,
                                          add_special_tokens=True)

        # truncation of tokens
        input_ids = input_ids[:self.max_tokens]

        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1]*len(input_ids)

        tensor_embeddings = torch.tensor(padded_input_ids).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        return img, mask, tensor_embeddings, attention_mask, sent_id

    def get_sent_raw(self, sent_id):
        return self.refer.sents[sent_id]["raw"]

    def get_image(self, sent_id):
        ref_id = self.refer.sent_to_ref[sent_id]
        ann_id = self.refer.ref_to_ann[ref_id]
        ann = self.refer.anns[ann_id]

        img_id = ann["image_id"]
        img_name = self.refer.imgs[img_id]["file_name"]
        img = Image.open(self.image_root + img_name)

        return img
