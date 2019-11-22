from Tasks.AbstractTask import AbstractTask
from Datasets.IrisDataset import IrisDataset
from Algorithms.kmeans import Kmeans

class ValidateKmeansTask(AbstractTask):
    def __init__(self):
        self.kmeans = Kmeans(k=3,seed=3)

    def _load_data(self):
        self.iris_ds = IrisDataset()
        self.iris_ds.load()
        self.iris_df = self.iris_ds.df

    def _preprocess_data(self):
        pass

    def _fit(self):
        self.kmeans.fit(data=self.iris_df,iterations=10)

    def execute(self):
        self._load_data()
        self._fit()
        self.centroids_samples = self.kmeans.centroid_sample_mapping
        print('centroids ditribution: {}'.format([len(samples) for _, samples in self.centroids_samples.items()]))
        self.score = self.kmeans.score
        print('score: {}'.format(self.score))

if __name__ == "__main__":
    task = ValidateKmeansTask()
    task.execute()