from Tasks.Task import AbstractTask
from Datasets.ImageDataset import ImageDataset
from skimage.transform import rescale, resize
from sklearn.cluster import KMeans
from Helpers.ImageHelper import ImageHelper
import sys

class ImageSegmentationKmeansTask(AbstractTask):
    def __init__(self):
        self.images = list()
        self.image_shapes = list()
        self.image_helper = ImageHelper()

    def _load_data(self):
        self.image_dataset = ImageDataset()
        self.image_dataset.load(self.path)
        self.raw_images = self.image_dataset.dataset

    def _resize_image(self, image, resize_factor):
        resized_image = resize(image, (image.shape[0] // resize_factor, image.shape[1] // resize_factor),
                               anti_aliasing=True)
        return resized_image

    def _preprocess_data(self, resize_factor):
        for image in self.raw_images:
            resized_image = self._resize_image(image, resize_factor)

            self.image_shapes.append(resized_image.shape)
            processed_image = resized_image.reshape(-1, 3)
            self.images.append(processed_image)


    def _fit(self,clusters,image):
        self.kmeans = KMeans(clusters, random_state=0)
        self.kmeans.fit(image)

    def execute(self,path,clusters,resize_factor):
        self.path = path

        self._load_data()
        self._preprocess_data(resize_factor=8)

        for index, image in enumerate(self.images):
            self._fit(clusters=5, image=image)
            x, y, z = self.image_shapes[index]  # z=3
            segmented_image = self.kmeans.cluster_centers_[self.kmeans.labels_].reshape(x, y, z)
            self.image_helper.show_image(segmented_image)

if __name__ == "__main__":
    task = ImageSegmentationKmeansTask()
    path = sys.argv[1]
    clusters = sys.argv[2]
    resize_factor = sys.argv[3]
    task.execute(path,clusters,resize_factor)