import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Annotated
from pydantic.fields import Field

CannyParamInt = Annotated[int, Field(gt=1, lt=254)]


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


def plot_contour(contour: np.ndarray, save_num: int):
    plt.figure()
    for i in range(len(contour)):
        contour_i = contour[i].reshape(-1, 2).T
        plt.plot(contour_i[0], contour_i[1], linewidth=0.5)
    plt.gca().set_aspect("equal")
    plt.tick_params(axis="both", which="both", direction="in")
    plt.xlim(0, 1226)
    plt.ylim(0, 1006)
    plt.savefig(f"testing/contour_{save_num}.png", dpi=300)
    plt.close()


# test.png
image_path = "testing/test.png"
for i in range(1, 254):
    contour = get_contour(image_path, canny_param_int=i)
    plot_contour(contour)
