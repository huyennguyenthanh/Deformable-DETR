


from pycocotools.coco import COCO

file = "../Poly/Train Dataset/annotations/annotation.json"
coco = COCO(file)

ids = list(sorted(coco.imgs.keys()))
print(ids[1])

id = coco.getAnnIds(1)
target = coco.loadAnns(id)
path = coco.loadImgs(id)[0]['filename']
print(path)

from PIL import Image
import matplotlib.pyplot as plt

import os  
source = "../Poly/Train Dataset/"
img = Image.open(os.path.join(source,path))
plt.imshow(img)