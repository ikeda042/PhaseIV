import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Annotated
from pydantic.fields import Field
import imageio

CannyParamInt = Annotated[int, Field(gt=1, lt=254)]
global_contour_sums = []


def get_contour(image_path: str, canny_param_int: CannyParamInt) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, canny_param_int, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(thresh, 0, 150)
    contours_raw, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    # # Filter out contours with small area
    contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours_raw))
    return contours


def plot_contour(contour: np.ndarray, save_path: str):
    plt.figure()
    for i in range(len(contour)):
        contour_i = contour[i].reshape(-1, 2).T
        plt.plot(contour_i[0], contour_i[1], linewidth=0.5)
    plt.gca().set_aspect("equal")
    plt.tick_params(axis="both", which="both", direction="in")
    plt.xlim(0, 1226)
    plt.ylim(0, 1006)
    number = save_path.split("_")[-1].split(".")[0]
    # put number on the upper right corner
    plt.text(
        0.95, 0.95, number, ha="center", va="center", transform=plt.gca().transAxes
    )
    plt.savefig(save_path, dpi=300)
    plt.close()
    plt.clf()
    # contour area sum
    global_contour_sums.append(sum([cv2.contourArea(c) for c in contour]))


image_path = "testing/test2.png"
for i in range(1, 254):
    contour = get_contour(image_path, canny_param_int=i)
    plot_contour(contour, f"testing/contour_{i}.png")
for i in range(1, 254):
    fig = plt.figure()
    print(range(i), global_contour_sums[:i])
    plt.plot(range(i), global_contour_sums[:i], marker="o")
    plt.xlabel("Canny Threshold")
    plt.ylabel("Area")
    plt.xlim(0, 254)
    plt.savefig(f"testing/contour_sum_{i}.png", dpi=300)

images = []
for i in range(1, 254):
    images.append(imageio.imread(f"testing/contour_{i}.png"))
imageio.mimsave("testing/contour.gif", images, loop=0)

images = []
for i in range(1, 254):
    images.append(imageio.imread(f"testing/contour_sum_{i}.png"))
imageio.mimsave("testing/contour_sum.gif", images, loop=0)
