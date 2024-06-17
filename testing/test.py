import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Annotated
from pydantic.fields import Field
from concurrent.futures import ThreadPoolExecutor
import imageio
from tqdm import tqdm

CannyParamInt = Annotated[int, Field(gt=1, lt=254)]
global_contour_sums = []


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
        plt.plot(contour_i[0], contour_i[1], linewidth=0.5)
    plt.gca().set_aspect("equal")
    plt.tick_params(axis="both", which="both", direction="in")
    plt.xlim(0, 1226)
    plt.ylim(0, 1006)
    plt.text(
        0.95, 0.95, str(number), ha="center", va="center", transform=plt.gca().transAxes
    )
    plt.savefig(save_path, dpi=300)
    plt.close()


def process_image(image, canny_param_int: CannyParamInt):
    contours = get_contour(image, canny_param_int)
    contour_sum = sum([cv2.contourArea(c) for c in contours])
    return canny_param_int, contours, contour_sum


def plot_contour_sum(global_contour_sums, i):
    plt.figure()
    plt.plot(range(1, i + 1), global_contour_sums[:i], marker="o")
    plt.xlabel("Canny Threshold")
    plt.ylabel("Area")
    plt.xlim(0, 254)
    plt.savefig(f"testing/contour_sum_{i}.png", dpi=300)
    plt.close()


def combine_images(i):
    img1 = cv2.imread(f"testing/contour_{i}.png")
    img2 = cv2.imread(f"testing/contour_sum_{i}.png")
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    combined = np.hstack((img1, img2))
    cv2.imwrite(f"testing/combined_{i}.png", combined)


image_path = "testing/test2.png"
image = cv2.imread(image_path)

# 並列処理のためのスレッドプールを作成
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(
        tqdm(executor.map(lambda i: process_image(image, i), range(1, 254)), total=253)
    )

# 結果を保存し、プロットをメインスレッドで作成
for canny_param_int, contours, contour_sum in results:
    plot_contour(contours, f"testing/contour_{canny_param_int}.png", canny_param_int)
    global_contour_sums.append(contour_sum)

for i in tqdm(range(1, 254)):
    plot_contour_sum(global_contour_sums, i)

with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(combine_images, range(1, 254)), total=253))

images = [imageio.imread(f"testing/combined_{i}.png") for i in tqdm(range(1, 254))]
imageio.mimsave("testing/contour_sum.gif", images, loop=0)
