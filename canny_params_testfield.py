import imageio
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Annotated
from pydantic.fields import Field

CannyParamInt = Annotated[int, Field(gt=1, lt=254)]
