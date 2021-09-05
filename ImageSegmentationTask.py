from ImageDataset import ImageDataset
from skimage.transform import  resize
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kmeans import Kmeans
import numpy as np


def show_image(image, image_name):
    plt.figure()
    plt.imshow(image)
    plt.savefig("{}.jpg".format(image_name))
    plt.show()


class ImageSegmentationTask:
    def __init__(self):
        self.images = list()
        self.image_shapes = list()

    def _load_data(self, path):
        self.image_dataset = ImageDataset(path)
        self.image_dataset.load()
        self.raw_images = self.image_dataset.get_dataset()

    @staticmethod
    def _resize_image(image, resize_factor):
        resized_image = resize(image, (image.shape[0] // resize_factor, image.shape[1] // resize_factor),
                               anti_aliasing=True)
        return resized_image

    def _preprocess_data(self, resize_factor):
        for image in self.raw_images:
            resized_image = self._resize_image(image, resize_factor)

            self.image_shapes.append(resized_image.shape)
            processed_image = resized_image.reshape(-1, 3)
            self.images.append(processed_image)

    def _fit(self, clusters, image):
        # self.kmeans = KMeans(clusters, random_state=0)
        kmeans = Kmeans(clusters, delta=0.001, seed=0)
        centroids, centroid_sample_mapping, sample_centroid_mapping = kmeans.fit(image, iterations=100)
        return centroids, centroid_sample_mapping, sample_centroid_mapping

    def execute(self, path, clusters, resize_factor):
        self._load_data(path)
        self._preprocess_data(resize_factor=resize_factor)

        for index, image in enumerate(self.images):
            centroids, centroid_sample_mapping, sample_centroid_mapping = self._fit(clusters=clusters, image=image)
            x, y, z = self.image_shapes[index]  # z=3
            segmented_image = np.asarray([np.asarray(centroids)[label] for label in sample_centroid_mapping.values()]).reshape(x, y, z)
            show_image(segmented_image, index)


