import cv2
import os

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
        if cv2.contourArea(contour) > 30:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame, f"Frame: {i}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    combined = cv2.hconcat([frame, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)])

    cv2.imshow("Combined", combined)

    if cv2.waitKey(int(1)) & 0xFF == 27:
        break

cv2.destroyAllWindows()
