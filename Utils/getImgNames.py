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


annotation_file = "datasets/refcoco/annotations.json"
dataset_anns = json.load(open(annotation_file, "r"))

ref_file = "datasets/refcoco/refs(unc).p"
dataset_refs = pickle.load(open(ref_file, "rb"))

out_refs = []
out_anns = {}
out_anns["info"] = dataset_anns["info"]
out_anns["categories"] = dataset_anns["categories"]
out_anns["licenses"] = dataset_anns["licenses"]

out_anns["images"] = []
out_anns["annotations"] = []

anns = {}
for ann in dataset_anns["annotations"]:
    anns[ann["id"]] = ann
imgs = {}
for img in dataset_anns["images"]:
    imgs[img["id"]] = img


file_names = set()

for i in range(len(dataset_refs)):
    ref = dataset_refs[i]
    out_refs.append(ref)

    ann_id = ref["ann_id"]
    ann = anns[ann_id]
    out_anns["annotations"].append(ann)

    img_id = ref["image_id"]
    img = imgs[img_id]
    file_name = img["file_name"]

    if file_name not in file_names:
        print(file_name)
        file_names.add(file_name)
