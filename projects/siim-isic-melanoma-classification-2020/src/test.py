import zipfile
from PIL import Image
f = zipfile.ZipFile('/data/kaggle/siim-isic-melanoma-classification.zip', 'r')
img_buf = f.open('jpeg/test/ISIC_0963827.jpg', 'r')
img = Image.open(img_buf)
img.save('img.jpg')