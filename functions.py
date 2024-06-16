import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_contour(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # use canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]


def plot_contour(contour: np.ndarray):
    plt.figure()
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], "r")
    plt.savefig("contour.png")
