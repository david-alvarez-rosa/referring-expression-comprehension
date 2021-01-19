"""
This file creates a smaller dataset! It's super useful.
"""

import json
import pickle
import numpy as np
import copy
import itertools
import os
from collections import defaultdict


annotation_file = 'data/instances.json'
dataset_anns = json.load(open(annotation_file, 'r'))

ref_file = 'data/refs(unc).p'
dataset_refs = pickle.load(open(ref_file, 'rb'))

print(dataset_refs[0].keys())


out_refs = []
out_anns = {}
out_anns['info'] = dataset_anns['info']
out_anns['categories'] = dataset_anns['categories']
out_anns['licenses'] = dataset_anns['licenses']

out_anns['images'] = []
out_anns['annotations'] = []

anns = {}
for ann in dataset_anns['annotations']:
    anns[ann['id']] = ann
imgs = {}
for img in dataset_anns['images']:
    imgs[img['id']] = img


for i in range(10):
    ref = dataset_refs[i]
    out_refs.append(ref)

    ann_id = ref['ann_id']
    ann = anns[ann_id]
    out_anns['annotations'].append(ann)

    img_id = ref['image_id']
    img = imgs[img_id]
    out_anns['images'].append(img)

with open("out_anns.json", "w") as out_anns_file:
    json.dump(out_anns, out_anns_file)
with open("out_refs.p", "wb") as out_refs_file:
    pickle.dump(out_refs, out_refs_file)
