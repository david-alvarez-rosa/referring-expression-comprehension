#!/bin/bash


curl -i -X POST \
     -d imgMethod=gallery \
     -d referringExpression="man right behind hey" \
     -d imgSrc=/home/david/Documents/UPC/Cuatrimestre\ 9/Bachelor\'s\ Thesis/Dataset/refcoco/images/COCO_train2014_000000578567.jpg \
     -d imgSrc=/home/david/Documents/UPC/Cuatrimestre\ 9/Bachelor\'s\ Thesis/test.jpg \
     -d debug=true \
     http://51.75.163.95/api/comprehend.php
     # http://localhost/api/comprehend.php
