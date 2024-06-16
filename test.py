import numpy as np
import matplotlib.pyplot as plt
from stardist import random_label_cmap
from stardist.models import StarDist2D
from skimage.io import imread
from skimage.color import rgb2gray
from stardist.plot import render_label

# カラーマップの設定
lbl_cmap = random_label_cmap()

# 事前学習済みモデルのロード
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# 画像の読み込み
image = imread('path_to_your_image.png')
gray_image = rgb2gray(image)

# オブジェクトの検出
labels, details = model.predict_instances(gray_image)

# 結果のプロット
plt.figure(figsize=(8, 8))
plt.subplot(121)
plt.imshow(gray_image, cmap='gray')
plt.title('Input Image')

plt.subplot(122)
plt.imshow(render_label(labels, img=gray_image))
plt.title('Detected Objects')
plt.show()
