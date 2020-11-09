from cocotools.referme import REFER
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


annFile = 'data/instances.json'
refFile = 'data/refs(unc).p'

refer = REFER(annFile, refFile)

refIds = refer.getRefIds()
refId = refIds[np.random.randint(0, len(refIds))]
ref = refer.loadRefs(refId)[0]
print(ref)

imgId = ref['image_id']
img = refer.loadImgs(imgId)[0]

# Get image.
I = io.imread('/home/david/Documents/UPC/Cuatrimestre 9/cocoapi/coco/train2014/{}'.format(img['file_name']))
I = np.asarray(I)
print(I.shape)
plt.imshow(I)
plt.axis('off')
plt.show()

# Show the category.
catId = ref['category_id']
print(refer.cats[catId])
annId = ref['ann_id']
ann = refer.loadAnns(annId)[0]
mask = refer.annToMask(ann)
plt.imshow(mask)
plt.show()

# Get sentences.
for sent in ref['sentences']:
    print(sent['sent'])

# Show annotation, built-in method.
plt.imshow(I)
plt.axis('off')
refer.showAnns([ann])
