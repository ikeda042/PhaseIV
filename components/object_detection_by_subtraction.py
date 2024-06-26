import cv2
import os
import imageio

fgbg = cv2.createBackgroundSubtractorMOG2()

num_images = len(os.listdir("sample_images/sample_tiff"))
frames = []
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
        if cv2.contourArea(contour) > 200:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            center_x = x + w // 2
            center_y = y + h // 2

            cv2.line(
                frame,
                (center_x - 10, center_y),
                (center_x + 10, center_y),
                (0, 0, 255),
                2,
            )
            cv2.line(
                frame,
                (center_x, center_y - 10),
                (center_x, center_y + 10),
                (0, 0, 255),
                2,
            )

    cv2.putText(
        frame, f"Frame: {i}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    combined = cv2.hconcat([frame, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)])

    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    frames.append(combined_rgb)

imageio.mimsave("timelapse_sub.gif", frames, duration=0.1, loop=0)
