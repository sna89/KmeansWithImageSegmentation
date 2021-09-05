import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd


def plot_elbow(y):
    plt.plot(1 + np.arange(len(y)), y)
    plt.ylabel('some numbers')
    plt.show()


class Kmeans():
    def __init__(self, k, delta, seed=None):
        self.k = k
        self.delta = delta
        self.centroids = []

        if seed:
            np.random.seed(seed)

    def _initialize_centroids(self, data, plus_plus_init=True):
        if plus_plus_init:
            self.centroids = self._kmeans_plus_plus(data)
        else:
            self.centroids = np.random.permutation(data)[:self.k]

    def _kmeans_plus_plus(self, data):
        # Algorithm for choosing initial values for centroids
        centroids = []

        for i in range(self.k):
            last_appended_centroid = self._choose_next_centroid_in_initialization(data, centroids)
            centroids.append(last_appended_centroid)

        return centroids

    @staticmethod
    def _choose_next_centroid_in_initialization(data, chosen_centroids=[]):
        num_samples = data.shape[0]

        if chosen_centroids:
            dist_from_centroids = None
            if isinstance(data, pd.DataFrame):
                dist_from_centroids = data.apply(Kmeans._distance_from_centroid, axis=1, args=(chosen_centroids,))
            elif isinstance(data, np.ndarray):
                dist_from_centroids = np.asarray([Kmeans._distance_from_centroid(data_point, chosen_centroids) for data_point in data])
            normalized_distance_list = dist_from_centroids / np.sum(dist_from_centroids)
            centroid_idx = np.random.choice(num_samples, size=1, replace=False, p=normalized_distance_list)

        else:
            centroid_idx = np.random.randint(low=0, high=num_samples, size=1)

        centroid = None
        if isinstance(data, pd.DataFrame):
            centroid = data.iloc[centroid_idx].values[0]
        elif isinstance(data, np.ndarray):
            centroid = data[centroid_idx]

        return centroid

    def _update_centroids(self, centroid_sample_mapping):
        for centroid_index, samples in centroid_sample_mapping.items():
            self.centroids[centroid_index] = np.mean(samples, axis=0)

    def _choose_centroid(self, sample):
        centroids_distance_from_record = list(
            map(lambda centroid: distance.euclidean(centroid, sample), self.centroids))
        centroid = np.argmin(centroids_distance_from_record)
        return centroid

    def _assign_samples_to_centroids(self, data):
        centroid_sample_mapping = dict()
        sample_centroid_mapping = dict()

        for index in range(self.k):
            centroid_sample_mapping[index] = list()

        if isinstance(data, pd.DataFrame):
            for sample_index, sample_features in data.iterrows():
                centroid = self._choose_centroid(sample_features)
                centroid_sample_mapping[centroid].append(sample_features)
                sample_centroid_mapping[sample_index] = centroid
        elif isinstance(data, np.ndarray):
            for sample_index, sample_features in enumerate(data):
                centroid = self._choose_centroid(sample_features)
                centroid_sample_mapping[centroid].append(sample_features)
                sample_centroid_mapping[sample_index] = centroid

        return centroid_sample_mapping, sample_centroid_mapping

    def _calc_score(self, centroid_sample_mapping):
        score = 0
        for centroid, samples in centroid_sample_mapping.items():
            score += sum(list(map(lambda x: distance.euclidean(x, self.centroids[centroid]), samples)))
        return score

    def fit(self, data, iterations):
        self._initialize_centroids(data)

        score = None
        last_score = None
        centroid_sample_mapping = None
        sample_centroid_mapping = None

        for i in range(iterations):
            centroid_sample_mapping, sample_centroid_mapping = self._assign_samples_to_centroids(data)
            self._update_centroids(centroid_sample_mapping)

            score = self._calc_score(centroid_sample_mapping)
            if last_score and (last_score - score) < self.delta:
                break
            last_score = score

        return self.centroids, centroid_sample_mapping, sample_centroid_mapping

    @property
    def get_centroids(self):
        return self.centroids

    @staticmethod
    def _distance_from_centroid(row, chosen_centroids):
        if hasattr(row, "values"):
            distance_from_centroids = list(map(lambda centroid: distance.euclidean(row.values, centroid),
                                               chosen_centroids))
        else:
            distance_from_centroids = list(map(lambda centroid: distance.euclidean(row, centroid),
                                               chosen_centroids))

        min_distance_from_centroids = min(distance_from_centroids)
        return min_distance_from_centroids
