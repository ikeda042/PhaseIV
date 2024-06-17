import imageio
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Annotated
from pydantic.fields import Field

CannyParamInt = Annotated[int, Field(gt=1, lt=254)]


def get_contour(image, canny_param_int: CannyParamInt) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, canny_param_int, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(thresh, 0, 150)
    contours_raw, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [c for c in contours_raw if cv2.contourArea(c) >= 300]
    return contours


def plot_contour(contour: np.ndarray, save_path: str, number: int):
    plt.figure()
    for i in range(len(contour)):
        contour_i = contour[i].reshape(-1, 2).T
        plt.plot(contour_i[0], contour_i[1], linewidth=2)
    plt.gca().set_aspect("equal")
    plt.tick_params(axis="both", which="both", direction="in")
    plt.xlim(0, 1226)
    plt.ylim(0, 1006)
    plt.text(
        0.95, 0.95, str(number), ha="center", va="center", transform=plt.gca().transAxes
    )
    plt.savefig(save_path, dpi=300)
    plt.close()
    plt.clf()


def process_image(image, canny_param_int: CannyParamInt):
    contours = get_contour(image, canny_param_int)
    contour_sum = sum([cv2.contourArea(c) for c in contours])
    return canny_param_int, contours, contour_sum
