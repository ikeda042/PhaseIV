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
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

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
    return lower, upper, contour_sum

def plot_heatmap(contour_sums, lower, upper, max_lower, max_upper):
    plt.figure(figsize=(10, 8))
    sns.heatmap(contour_sums, cmap="YlGnBu", cbar=True, xticklabels=50, yticklabels=50)
    plt.xlabel('Canny Upper Threshold')
    plt.ylabel('Canny Lower Threshold')
    plt.title(f'Canny Thresholds: Lower={lower}, Upper={upper}')
    plt.savefig(f"tmp/heatmap_{lower}_{upper}.png")
    plt.close()

def plot_worker():
    while True:
        task = plot_queue.get()
        if task is None:
            break
        plot_heatmap(*task)
        plot_queue.task_done()

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

plot_queue = queue.Queue()
plot_thread = threading.Thread(target=plot_worker)
plot_thread.start()

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_image, image, lower, upper) for lower in range(1, max_threshold) for upper in range(lower + 1, max_threshold + 1)]
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        lower, upper, contour_sum = future.result()
        contour_sums[lower, upper] = contour_sum
        plot_queue.put((contour_sums, lower, upper, max_threshold, max_threshold))

plot_queue.put(None)
plot_thread.join()

images = [imageio.imread(f"tmp/heatmap_{lower}_{upper}.png") for lower in range(1, max_threshold) for upper in range(lower + 1, max_threshold + 1)]
imageio.mimsave("result_heatmap.gif", images, loop=0)
