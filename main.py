import cv2 
import os
import random
from module1 import *
from module2 import *
from module3 import *
from module4 import *
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt 
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


if os.path.isdir('data') is False:
   setup1()


collect_positive_anchor()
