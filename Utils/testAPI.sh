#!/bin/bash


curl -i -X POST \
     -d imgMethod=gallery \
     -d referringExpression="man right behind" \
     -d imgSrc=datasets/refcoco/images/COCO_train2014_000000578567.jpg \
     http://localhost/prueba.php
