from ImageSegmentationTask import ImageSegmentationTask
import os
from IrisDataset import IrisDataset
from kmeans import Kmeans, plot_elbow


if __name__ == "__main__":
    iris_ds = IrisDataset().get_data()
    scores = []
    for k in range(1, 11):
        kmeans = Kmeans(k, 0.001)
        centroid_sample_mapping, score = kmeans.fit(iris_ds, iterations=100)
        scores.append(score)
    plot_elbow(scores)

    segmentation = ImageSegmentationTask()
    path = os.path.join(os.getcwd(), "Images")
    segmentation.execute(path, clusters=6, resize_factor=8)
