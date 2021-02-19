#!/usr/bin/env python


"""Interface for accessing the Microsoft Refer ann_dataset.

This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
Refer      - Refer api class
get_ref_ids  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
get_img_ids  - get image ids that satisfy given filter conditions.
get_cat_ids  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref"s bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
get_mask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import json
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from pycocotools import mask as mask_utils
from collections import defaultdict


__author__ = "Rob Knight, Gavin Huttley, and Peter Maxwell"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Rob Knight", "Peter Maxwell", "Gavin Huttley",
               "Matthew Wakefield"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Production"


def _is_array_like(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class Refer:
    """Description of class.

    Longer description of class.
    Longer description of class.
    """

    def __init__(self, ann_file, ref_file):
        """Short description.

        provide data_root folder which contains refclef, refcoco, refcoco+ and
        refcocog also provide ann_dataset name and splitBy information e.g.,
        ann_dataset = "refcoco", splitBy = "unc"
        """

        # TODO: remove this
        self.image_dir = ("/home/david/Documents/UPC/Cuatrimestre 9/"
                          "Bachelor's Thesis/datasets/refcoco/images")

        print("Loading annotations into memory...")
        tic = time.time()
        ann_dataset = json.load(open(ann_file, "r"))
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self.ann_dataset = ann_dataset

        print("Loading referring expressions into memory...")
        tic = time.time()
        ref_dataset = pickle.load(open(ref_file, "rb"))
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self.ref_dataset = ref_dataset

        self.create_index()

    def create_index(self):
        """asdf

        create sets of mapping
        1)  refs:         {ref_id: ref}
        2)  anns:         {ann_id: ann}
        3)  imgs:         {image_id: image}
        4)  cats:         {category_id: category_name}
        5)  sents:        {sent_id: sent}
        6)  img_to_refs:    {image_id: refs}
        7)  img_to_anns:    {image_id: anns}
        8)  ref_to_ann:     {ref_id: ann}
        9)  ann_to_ref:     {ann_id: ref}
        10) cat_to_refs:    {category_id: refs}
        11) sentToRef:    {sent_id: ref}
        12) sent_to_tokens: {sent_id: tokens}
        """

        print("Creating index...")
        tic = time.time()

        # Fetch info from annotation dataset.
        anns, imgs, cats = {}, {}, {}
        cat_to_imgs, img_to_anns = defaultdict(list), defaultdict(list)
        for ann in self.ann_dataset["annotations"]:
            anns[ann["id"]] = ann
            img_to_anns[ann["image_id"]].append(ann["id"])
            cat_to_imgs[ann["category_id"]].append(ann["image_id"])
        for img in self.ann_dataset["images"]:
            imgs[img["id"]] = img
        for cat in self.ann_dataset["categories"]:
            cats[cat["id"]] = cat["name"]

        # Fetch info from referring dataset.
        refs, sents = {}, {}
        ann_to_ref, ref_to_ann = {}, {}
        ref_to_sents = defaultdict(list)
        sent_to_tokens, sent_to_ref = {}, {}
        for ref in self.ref_dataset:
            refs[ref["ref_id"]] = ref
            ann_to_ref[ref["ann_id"]] = ref["ref_id"]
            ref_to_ann[ref["ref_id"]] = ref["ann_id"]
            for sent in ref["sentences"]:
                sents[sent["sent_id"]] = sent
                sent_to_tokens[sent["sent_id"]] = sent["tokens"]
                ref_to_sents[ref["ref_id"]].append(sent["sent_id"])
                sent_to_ref[sent["sent_id"]] = ref["ref_id"]

        print("Done (t={:0.2f}s)".format(time.time() - tic))

        # Set attributes.
        self.cats = cats
        self.imgs = imgs
        self.anns = anns
        self.refs = refs
        self.sents = sents
        self.cat_to_imgs = cat_to_imgs
        self.img_to_anns = img_to_anns
        self.ann_to_ref = ann_to_ref
        self.ref_to_ann = ref_to_ann
        self.ref_to_sents = ref_to_sents
        self.sent_to_ref = sent_to_ref
        self.sent_to_tokens = sent_to_tokens

    def ann_info(self):
        """Prints information about the annotation file."""

        for key, value in self.ann_dataset["info"].items():
            print("{}: {}".format(key, value))

    def get_sent_ids(self,
                     cat_names=None,
                     cat_ids=None,
                     sup_names=None,
                     img_ids=None,
                     area_range=None,
                     is_crowd=None,
                     ann_ids=None,
                     split=None,
                     ref_ids=None,
                     sent_ids=None):
        """Get ann ids that satisfy given filter conditions.

        Args:
          cat_names:
            A list of strings specifying cat names or None if filter is
            deactivated. A single string will also work.
          cat_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.
          sup_names:
            A list of strings specifying supercategory names or None if filter
            is deactivated. A single string will also work.
          img_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.
          area_range:
            A list of two integers specifying area range (e.g. [0 inf]) or None
            if filter is deactivated.
          is_crowd:
            A boolean specifying crowd label or None if filter is deactivated.
          ann_ids:
            A list of integers specifying ann ids or None if filter is
            deactivated. A single integer will also work.
          split:
            A string specifying split label (train/val/test) or None if filter
            is deactivated.
          ref_ids:
            A list of integers specifying ann ids or None if filter is
            deactivated. A single integer will also work.
          sent_ids:
            A list of integers specifying ann ids or None if filter is
            deactivated. A single integer will also work.

        Returns:
          A list of integers specifying the sent ids.
        """

        ref_ids = self.get_ref_ids(cat_names=cat_names,
                                   cat_ids=cat_ids,
                                   sup_names=sup_names,
                                   img_ids=img_ids,
                                   area_range=area_range,
                                   is_crowd=is_crowd,
                                   ann_ids=ann_ids,
                                   split=split,
                                   ref_ids=ref_ids)

        ids = []
        for ref_id in ref_ids:
            ids += self.ref_to_sents[ref_id]
        if sent_ids is not None:
            sent_ids = sent_ids if _is_array_like(sent_ids) else [sent_ids]
            ids = [id_ for id_ in ids if id_ in sent_ids]
        return ids

    def get_ref_ids(self,
                    cat_names=None,
                    cat_ids=None,
                    sup_names=None,
                    img_ids=None,
                    area_range=None,
                    is_crowd=None,
                    ann_ids=None,
                    split=None,
                    ref_ids=None):
        """Get ann ids that satisfy given filter conditions.

        Args:
          cat_names:
            A list of strings specifying cat names or None if filter is
            deactivated. A single string will also work.
          cat_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.
          sup_names:
            A list of strings specifying supercategory names or None if filter
            is deactivated. A single string will also work.
          img_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.
          area_range:
            A list of two integers specifying area range (e.g. [0 inf]) or None
            if filter is deactivated.
          is_crowd:
            A boolean specifying crowd label or None if filter is deactivated.
          ann_ids:
            A list of integers specifying ann ids or None if filter is
            deactivated. A single integer will also work.
          split:
            A string specifying split label (train/val/test) or None if filter
            is deactivated.
          ref_ids:
            A list of integers specifying ann ids or None if filter is
            deactivated. A single integer will also work.

        Returns:
          A list of integers specifying the ref ids.
        """

        ann_ids = self.get_ann_ids(cat_names=cat_names,
                                   cat_ids=cat_ids,
                                   sup_names=sup_names,
                                   img_ids=img_ids,
                                   area_range=area_range,
                                   is_crowd=is_crowd,
                                   ann_ids=ann_ids)

        refs = [self.refs[self.ann_to_ref[ann_id]] for ann_id in ann_ids
                if ann_id in self.ann_to_ref]
        if split is not None:
            refs = [ref for ref in refs if ref["split"] == split]
        if ref_ids is not None:
            ref_ids = ref_ids if _is_array_like(ref_ids) else [ref_ids]
            refs = [ref for ref in refs if ref["id"] in ref_ids]

        ids = [ref["ref_id"] for ref in refs]
        return ids

    def get_ann_ids(self,
                    cat_names=None,
                    cat_ids=None,
                    sup_names=None,
                    img_ids=None,
                    area_range=None,
                    is_crowd=None,
                    ann_ids=None):
        """Get ann ids that satisfy given filter conditions.

        Args:
          cat_names:
            A list of strings specifying cat names or None if filter is
            deactivated. A single string will also work.
          cat_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.
          sup_names:
            A list of strings specifying supercategory names or None if filter
            is deactivated. A single string will also work.
          img_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.
          are_range:
            A list of two integers specifying area range (e.g. [0 inf]) or None
            if filter is deactivated.
          is_crowd:
            A boolean specifying crowd label or None if filter is deactivated.
          ann_ids:
            A list of integers specifying ann ids or None if filter is
            deactivated. A single integer will also work.

        Returns:
          A list of integers specifying the ann ids.
        """

        img_ids = self.get_img_ids(cat_names=cat_names,
                                   cat_ids=cat_ids,
                                   sup_names=sup_names,
                                   img_ids=img_ids)
        ann_ids_ = []
        for img_id in img_ids:
            ann_ids_ += self.img_to_anns[img_id]
        if area_range is not None:
            ann_ids_ = [ann_id for ann_id in ann_ids_
                        if self.anns[ann_id]["area"] > area_range[0]
                        and self.anns[ann_id]["area"] < area_range[1]]
        if is_crowd is not None:
            ann_ids_ = [ann_id for ann_id in ann_ids_
                       if self.anns[ann_id]["iscrowd"] == is_crowd]
        if ann_ids is not None:
            ann_ids = ann_ids if _is_array_like(ann_ids) else [ann_ids]
            ann_ids_ = [ann_id for ann_id in ann_ids_ if ann_id in ann_ids]

        return ann_ids_

    def get_img_ids(self,
                    cat_names=None,
                    cat_ids=None,
                    sup_names=None,
                    img_ids=None):
        """Get img ids that satisfy given filter conditions.

        Args:
          cat_names:
            A list of strings specifying cat names or None if filter is
            deactivated. A single string will also work.
          cat_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.
          sup_names:
            A list of strings specifying supercategory names or None if filter
            is deactivated. A single string will also work.
          img_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.

        Returns:
          A list of integers specifying the img ids.
        """

        cat_ids = self.get_cat_ids(cat_names=cat_names,
                                   sup_names=sup_names,
                                   cat_ids=cat_ids)
        ids = []
        for cat_id in cat_ids:
            ids += self.cat_to_imgs[cat_id]
        if img_ids is not None:
            img_ids = img_ids if _is_array_like(img_ids) else [img_ids]
            ids = [id_ for id_ in ids if id_ in img_ids]

        return list(set(ids))

    def get_cat_ids(self, cat_names=None, sup_names=None, cat_ids=None):
        """Get cat ids that satisfy given filter conditions.

        Args:
          cat_names:
            A list of strings specifying cat names or None if filter is
            deactivated. A single string will also work.
          sup_names:
            A list of strings specifying supercategory names or None if filter
            is deactivated. A single string will also work.
          cat_ids:
            A list of integers specifying cat ids or None if filter is
            deactivated. A single integer will also work.

        Returns:
          A list of integers specifying the cat ids.
        """

        cats = self.ann_dataset["categories"]

        if cat_names is not None:
            cat_names = cat_names if _is_array_like(cat_names) else [cat_names]
            cats = [cat for cat in cats if cat["name"] in cat_names]
        if sup_names is not None:
            sup_names = sup_names if _is_array_like(sup_names) else [sup_names]
            cats = [cat for cat in cats if cat["supercategory"] in sup_names]
        if cat_ids is not None:
            cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]
            cats = [cat for cat in cats if cat["id"] in cat_ids]

        ids = [cat["id"] for cat in cats]
        return ids

    def load_sents(self, ids=None):
        """Load sents with the specified ids.

        Args:
          ids:
            A list of integers specifying the sent ids or None to load all
            sents. A single integer will also work.

        Returns:
          A list of sents for all the specfied ids, or a single sent if
          ids is a single integer.
        """

        if _is_array_like(ids):
            return [self.sents[id_] for id_ in ids]
        return self.sents[ids]

    def load_refs(self, ids=None):
        """Load refs with the specified ids.

        Args:
          ids:
            A list of integers specifying the sent ids or None to load all
            refs. A single integer will also work.

        Returns:
          A list of refs for all the specfied ids, or a single sent if ids is a
          single integer.
        """

        if _is_array_like(ids):
            return [self.refs[id_] for id_ in ids]
        return self.refs[ids]

    def load_anns(self, ids=None):
        """Load anns with the specified ids.

        Args:
          ids:
            A list of integers specifying the sent ids or None to load all
            anns. A single integer will also work.

        Returns:
          A list of anns for all the specfied ids, or a single sent if ids is a
          single integer.
        """

        if _is_array_like(ids):
            return [self.anns[id_] for id_ in ids]
        return self.anns[ids]

    def load_imgs(self, ids=None):
        """Load imgs with the specified ids.

        Args:
          ids:
            A list of integers specifying the sent ids or None to load all
            imgs. A single integer will also work.

        Returns:
          A list of imgs for all the specfied ids, or a single sent if ids is a
          single integer.
        """

        if _is_array_like(ids):
            return [self.imgs[id_] for id_ in ids]
        return self.imgs[ids]

    def load_cats(self, ids=None):
        """Load cats with the specified ids.

        Args:
          ids:
            A list of integers specifying the sent ids or None to load all
            cats. A single integer will also work.

        Returns:
          A list of cats for all the specfied ids, or a single sent if ids is a
          single integer.
        """

        if _is_array_like(ids):
            return [self.cats[id_] for id_ in ids]
        return self.cats[ids]

    def show_anns(self, anns, draw_bbox=False):
        """Display the specified annotations.

        Args:
          anns:
            List of ann to display. It will also work with a single ann.
          draw_bbow:
            A boolean specifying if the bounding box should be drawn.
        """

        anns = anns if _is_array_like(anns) else [anns]

        if "segmentation" in anns[0] or "keypoints" in anns[0]:
            dataset_type = "instances"
        elif "caption" in anns[0]:
            dataset_type = "captions"
        else:
            raise Exception("dataset_type not supported")
        if dataset_type == "instances":
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6 + 0.4).tolist()[0]
                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], list):
                        # polygon
                        for seg in ann["segmentation"]:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann["image_id"]]
                        if isinstance(ann["segmentation"]["counts"], list):
                            rle = mask_utils.frPyObjects([ann["segmentation"]],
                                                        t["height"],
                                                        t["width"])
                        else:
                            rle = [ann["segmentation"]]
                        m = mask_utils.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann["iscrowd"] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0])/255
                        if ann["iscrowd"] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m*0.5)))
                if "keypoints" in ann and isinstance(ann["keypoints"], list):
                    # turn skeleton into zero-based index
                    sks = np.array(
                        self.loadCats(ann["category_id"])[0]["skeleton"]
                    ) - 1
                    kp = np.array(ann["keypoints"])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(x[v > 0], y[v > 0], "o", markersize=8,
                             markerfacecolor=c, markeredgecolor="k",
                             markeredgewidth=2)
                    plt.plot(x[v > 1], y[v > 1], "o", markersize=8,
                             markerfacecolor=c, markeredgecolor=c,
                             markeredgewidth=2)

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann["bbox"]
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h],
                            [bbox_x + bbox_w, bbox_y + bbox_h],
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0,
                                alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor="none", edgecolors=color,
                                linewidths=2)
            ax.add_collection(p)
        elif dataset_type == "captions":
            for ann in anns:
                print(ann["caption"])

    def ann_to_RLE(self, ann):
        """Convert annotation to RLE.

        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)

        Args:
          ann:
            Annotation object.

        Returns:
          A numpy 2D array specifying the binary mask.
        """

        t = self.imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def ann_to_mask(self, ann):
        """Convert annotation to binary mask.

        Convert annotation which can be polygons, uncompressed RLE, or RLE to
        binary mask_utils.

        Args:
          ann:
            Annotation object.

        Returns:
          A numpy 2D array specifying the binary mask.
        """

        rle = self.ann_to_RLE(ann)
        m = mask_utils.decode(rle)
        return m
