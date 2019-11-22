from Datasets.Dataset import AbsractImageDataset
import os

class ImageDataset(AbsractImageDataset):
    def __init__(self):
        self.name = 'images'
        super(ImageDataset,self).__init__()

    def load(self,full_path):
        self.full_path = full_path
        for image_file in os.listdir(self.full_path):
            image = self._read_image(self.full_path, image_file)
            self.dataset.append(image)

    @property
    def get_dataset(self):
        return self.dataset




