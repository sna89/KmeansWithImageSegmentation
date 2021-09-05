import os
from skimage import io


class ImageDataset:
    def __init__(self, path):
        self.path = path
        self.data = []

    def load(self):
        for image_filename in os.listdir(self.path):
            image = io.imread(os.path.join(self.path, image_filename))
            self.data.append(image)

    def get_dataset(self):
        return self.data

