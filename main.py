import cv2 
import os
import random
import module1
import module2
import module3
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt 
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


if os.path.isdir('data') is False:
   module1.setup1()


module2.collect_positive_anchor()
