# import cv2
# import numpy as np
# import os


# def find_optimal_shift(image1, image2):
#     result = cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)
#     _, _, min_loc, max_loc = cv2.minMaxLoc(result)
#     dx, dy = max_loc[0], max_loc[1]
#     return dx, dy


# def shift_image(image, dx, dy):
#     rows, cols = image.shape
#     M = np.float32([[1, 0, dx], [0, 1, dy]])
#     shifted_image = cv2.warpAffine(image, M, (cols, rows))
#     return shifted_image


# def detect_changes(image1, image2, threshold):
#     dx, dy = find_optimal_shift(image1, image2)
#     shifted_image2 = shift_image(image2, dx, dy)

#     diff = cv2.absdiff(image1, shifted_image2)
#     _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

#     return mask


# num_files = len(os.listdir("sample_images/sample_tiff"))
# for i in range(num_files - 1):
#     image1 = cv2.imread(f"sample_images/sample_tiff/{i}.tif", cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(f"sample_images/sample_tiff/{i + 1}.tif", cv2.IMREAD_GRAYSCALE)

#     threshold = 30
#     mask = detect_changes(image1, image2, threshold)

#     cv2.imwrite(f"mask_{i}.png", mask)

import cv2
import os
from datetime import datetime, timedelta

fgbg = cv2.createBackgroundSubtractorMOG2()

num_images = len(os.listdir("sample_images/sample_tiff"))

for i in range(num_images):
    frame = cv2.imread(f"sample_images/sample_tiff/{i}.tif")
    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    fgmask = fgbg.apply(blurred)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 2:  # ノイズを除くために面積フィルターをかける
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 画面にフレーム数を表示
    cv2.putText(
        frame, f"Frame: {i}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # 結果を表示
    combined = cv2.hconcat([frame, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)])

    # Display the combined frame
    cv2.imshow("Combined", combined)

    # Wait for the specified time in milliseconds
    if cv2.waitKey(int(1)) & 0xFF == 27:
        break

cv2.destroyAllWindows()
