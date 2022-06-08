


# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
from torchvision.datasets.vision import VisionDataset


from pycocotools.coco import COCO

def PolypDataset(VisionDataset):
    def __init__(self, root, annFile):
        super(PolypDataset, self).__init__()

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "Train Dataset", root / "annotations" / f'annotations.json'),
        "val": (root / "Val Dataset", root / "annotations" / f'annotations.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = PolypDataset(img_folder, ann_file, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
