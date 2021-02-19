#!/bin/bash


curl -i -X POST \
     -d imgMethod=gallery \
     -d referringExpression="man right behind" \
     -d imgSrc=/home/david/Documents/UPC/Cuatrimestre\ 9/Bachelor\'s\ Thesis/Dataset/refcoco/images/COCO_train2014_000000578567.jpg \
     http://localhost/api/prueba.php
