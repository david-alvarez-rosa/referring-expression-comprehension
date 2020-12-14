"""Explanation.

TODO: explanation.

More text.
"""

import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import transformers
from refer.refer import REFER


class ReferDataset(data.Dataset):
    """TODO: docstring."""

    def __init__(self,
                 args,
                 input_size,
                 image_transforms=None,
                 target_transforms=None,
                 split="train",
                 eval_mode=False):

        self.classes = []
        self.input_size = input_size
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split

        ref_file = "datasets/refcoco/refs(unc).p"
        ann_file = "datasets/refcoco/annotations.json"
        self.refer = REFER(ann_file, ref_file)

        self.max_tokens = 20

        #ref_ids = self.refer.get_ref_ids(split=self.split)  # Set split in future
        ref_ids = self.refer.get_ref_ids()

        img_ids = self.refer.get_img_ids(ref_ids)

        all_imgs = self.refer.imgs
        # print(all_imgs)
        # self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode

        for ref_id in ref_ids:
            ref = self.refer.refs[ref_id]

            sentences_for_ref = []
            attentions_for_ref = []

            for sent in ref["sentences"]:
                sentence_raw = sent["raw"]
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw,
                                                  add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_ann_id = self.refer.ref_to_ann[this_ref_id]
        this_ann = self.refer.anns[this_ann_id]
        this_img_id = this_ann["image_id"]
        this_img = self.refer.imgs[this_img_id]

        IMAGE_DIR = "datasets/refcoco/images"
        img = Image.open(os.path.join(IMAGE_DIR, this_img["file_name"])).convert("RGB")

        ref = self.refer.load_refs(this_ref_id)
        this_sent_ids = ref["sent_ids"]

        ref_mask = self.refer.ann_to_mask(this_ann)
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:

            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)

        else:

            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask
