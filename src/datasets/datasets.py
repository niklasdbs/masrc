"""
Module contains data loading functionality for the Melbourne On-street Car
Parking Sensor Data - 2017 dataset.
"""
from enum import Enum



class DataSplit(Enum):
    """
    Describes the different splits of the data.
    """
    TRAINING = 0
    TEST = 1
    VALIDATION = 2