import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_contour(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(thresh, 0, 150)
    contours_raw, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    # # Filter out contours with small area
    # contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours_raw))
    return contours_raw


def plot_contour(contour: np.ndarray):
    plt.figure()
    for i in range(len(contour)):
        plt.scatter(contour[i][:, 0, 0], contour[i][:, 0, 1], s=1)
    plt.savefig("contour.png", dpi=300)
    plt.close()


# test.png
image_path = "test.png"
contour = get_contour(image_path)
plot_contour(contour)
print(contour)
