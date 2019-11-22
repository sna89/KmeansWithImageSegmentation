import pandas as pd
import numpy as np
import os
from skimage import io, transform
from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    def __init__(self):
        self.name = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_sample(self):
        pass

    @property
    def get_name(self):
        return self.name


class AbsractDataFrameDataset(AbstractDataset):
    def __init__(self):
        self.df = None

    @abstractmethod
    def load(self):
        pass

    @property
    def get_dataset(self):
        return self.df

    def get_sample(self):
        return self.df.iloc[0]


class AbsractImageDataset(AbstractDataset):
    def __init__(self):
        self.dataset = list()

    @abstractmethod
    def load(self):
        pass

    @property
    def get_dataset(self):
        return self.dataset

    def get_sample(self):
        return self.images[0]


    def _read_image(self,path,image_filename):
        image = io.imread(os.path.join(path, image_filename))
        return image








