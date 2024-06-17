import imageio
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Annotated
from pydantic.fields import Field
import os
import shutil

CannyParamInt = Annotated[int, Field(gt=1, lt=254)]

def get_contour(image, canny_param_lower: CannyParamInt, canny_param_upper: CannyParamInt) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, canny_param_lower, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(thresh, canny_param_lower, canny_param_upper)
    contours_raw, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [c for c in contours_raw if cv2.contourArea(c) >= 300]
    return contours

def process_image(image, lower: CannyParamInt, upper: CannyParamInt):
    contours = get_contour(image, lower, upper)
    contour_sum = sum([cv2.contourArea(c) for c in contours])
    return lower, upper, contours, contour_sum

def plot_heatmap(contour_sums, lower, upper, max_lower, max_upper):
    plt.figure(figsize=(10, 8))
    sns.heatmap(contour_sums, cmap="YlGnBu", cbar=True, xticklabels=50, yticklabels=50)
    plt.xlabel('Canny Upper Threshold')
    plt.ylabel('Canny Lower Threshold')
    plt.title(f'Canny Thresholds: Lower={lower}, Upper={upper}')
    plt.savefig(f"tmp/heatmap_{lower}_{upper}.png")
    plt.close()

############################################################################################################
image_path = "sample_images/cells_100x_large_scope.png"
############################################################################################################
try:
    os.mkdir("tmp")
except FileExistsError:
    shutil.rmtree("tmp")
    os.mkdir("tmp")

image = cv2.imread(image_path)
max_threshold = 255
contour_sums = np.zeros((max_threshold + 1, max_threshold + 1))

for lower in tqdm(range(1, max_threshold)):
    for upper in range(lower + 1, max_threshold + 1):
        _, _, _, contour_sum = process_image(image, lower, upper)
        contour_sums[lower, upper] = contour_sum
        plot_heatmap(contour_sums, lower, upper, max_threshold, max_threshold)

images = [imageio.imread(f"tmp/heatmap_{lower}_{upper}.png") for lower in range(1, max_threshold) for upper in range(lower + 1, max_threshold + 1)]
imageio.mimsave("result_heatmap.gif", images, loop=0)
