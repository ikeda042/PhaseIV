import imageio
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Annotated
from pydantic.fields import Field
import os
import shutil

CannyParamInt = Annotated[int, Field(gt=1, lt=254)]


def get_contour(
    image, canny_param_lower: CannyParamInt, canny_param_upper: CannyParamInt
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, canny_param_lower, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(thresh, canny_param_lower, canny_param_upper)
    contours_raw, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [c for c in contours_raw if cv2.contourArea(c) >= 300]
    return contours


def plot_contour(contour: np.ndarray, save_path: str, lower: int, upper: int):
    plt.figure()
    for i in range(len(contour)):
        contour_i = contour[i].reshape(-1, 2).T
        plt.plot(contour_i[0], contour_i[1], linewidth=2)
    plt.gca().set_aspect("equal")
    plt.tick_params(axis="both", which="both", direction="in")
    plt.xlim(0, 1226)
    plt.ylim(0, 1006)
    plt.text(
        0.95,
        0.95,
        f"{lower}-{upper}",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    plt.savefig(save_path, dpi=300)
    plt.close()
    plt.clf()


def process_image(image, lower: CannyParamInt, upper: CannyParamInt):
    contours = get_contour(image, lower, upper)
    contour_sum = sum([cv2.contourArea(c) for c in contours])
    return lower, upper, contours, contour_sum


def plot_contour_sum(global_contour_sums, lower, upper):
    plt.figure()
    plt.plot(
        range(1, len(global_contour_sums) + 1),
        global_contour_sums,
        marker="o",
        markersize=1,
    )
    plt.xlabel("Canny Threshold Combination Index")
    plt.ylabel("Area")
    plt.xlim(0, len(global_contour_sums) + 1)
    plt.ylim(-10, 16000)
    plt.savefig(f"tmp/contour_sum_{lower}_{upper}.png", dpi=300)
    plt.close()
    plt.clf()


############################################################################################################
image_path = "sample_images/cells_100x_large_scope.png"
############################################################################################################
try:
    os.mkdir("tmp")
except FileExistsError:
    shutil.rmtree("tmp")

image = cv2.imread(image_path)
global_contour_sums = []
index = 0

for lower in tqdm(range(1, 255)):
    for upper in range(lower + 1, 256):
        index += 1
        lower, upper, contours, contour_sum = process_image(image, lower, upper)
        global_contour_sums.append(contour_sum)
        plot_contour(contours, f"tmp/contour_{lower}_{upper}.png", lower, upper)
        plot_contour_sum(global_contour_sums, lower, upper)

for lower in range(1, 255):
    for upper in range(lower + 1, 256):
        img1 = cv2.imread(f"tmp/contour_{lower}_{upper}.png")
        img2 = cv2.imread(f"tmp/contour_sum_{lower}_{upper}.png")
        combined = np.concatenate([img1, img2], axis=1)
        cv2.imwrite(f"tmp/combined_{lower}_{upper}.png", combined)

images = [
    imageio.imread(f"tmp/combined_{lower}_{upper}.png")
    for lower in range(1, 255)
    for upper in range(lower + 1, 256)
]
imageio.mimsave("result.gif", images, loop=0)
