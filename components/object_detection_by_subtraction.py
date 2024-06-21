import cv2
import numpy as np
import os


def find_optimal_shift(image1, image2):
    result = cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
    dx, dy = max_loc[0], max_loc[1]
    return dx, dy


def shift_image(image, dx, dy):
    rows, cols = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(image, M, (cols, rows))
    return shifted_image


def detect_changes(image1, image2, threshold):
    dx, dy = find_optimal_shift(image1, image2)
    shifted_image2 = shift_image(image2, dx, dy)

    diff = cv2.absdiff(image1, shifted_image2)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return mask


num_files = len(os.listdir("sample_images/sample_tiff"))
for i in range(num_files - 1):
    image1 = cv2.imread(f"sample_images/sample_tiff/{i}.tif", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(f"sample_images/sample_tiff/{i + 1}.tif", cv2.IMREAD_GRAYSCALE)

    threshold = 30
    mask = detect_changes(image1, image2, threshold)

    cv2.imwrite(f"mask_{i}.png", mask)
