# PhaseIV

# Components 

## Param search component

* Finds the best parameter for canny edge detection for microscopic images.

[search_canny_param_component.py](components/search_canny_param_component.py)

**Raw image**

![](sample_images/cells_100x_large_scope.png)

**Total area fluctuation by param1 of the cv2.canny() method.**

![](docs_images/param_serach.gif)

## Cell division annotator (WIP)

[object_detection_by_subtraction.py](components/object_detection_by_subtraction.py)

* Detects dividing cells in a timelapse video. 

![](docs_images/timelapse_sub.gif)