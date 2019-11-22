import numpy as np
from scipy.spatial import distance
import random
import pandas as pd

class Kmeans():
    def __init__(self, k, seed=None):
        self.k = k

        if seed:
            np.random.seed(seed)

    def _initilize_centroids(self, plus_plus_init = True):
        if plus_plus_init:
            self.centroids = self._kmeans_plus_plus()
        else:
            self.centroids = np.random.permutation(data)[:self.k]

    def _kmeans_plus_plus(self):
        centroids = []
        init_centroid = last_appended_centroid = self._choose_next_centroid_in_initialization()
        centroids.append(init_centroid)

        for i in range(1, self.k):
            centroid = self._choose_next_centroid_in_initialization(last_appended_centroid)
            centroids.append(centroid)
            last_appended_centroid = centroid

        return centroids

    def _distance_from_centroid(self,row,centroid):
        return distance.euclidean(row.values,centroid)

    def _choose_next_centroid_in_initialization(self,last_appended_centroid=None):
        num_samples = self.data.shape[0]
        min_distance_list = np.repeat(np.iinfo(np.int32).max, num_samples)

        if last_appended_centroid is not None:
            dist_from_centroid = self.data.apply(self._distance_from_centroid, axis=1, args=(last_appended_centroid,))
            min_distance_list = np.where(np.less(dist_from_centroid, min_distance_list), dist_from_centroid, min_distance_list)
            normalized_distance_list = min_distance_list / np.sum(min_distance_list)
            centroid_idx = np.random.choice(num_samples, size=1, replace=False, p=normalized_distance_list)
            centroid = self.data.iloc[centroid_idx].values[0]

        else:
            init_centroid_idx = np.random.uniform(low=0, high=num_samples, size=1)
            centroid = self.data.iloc[init_centroid_idx].values[0]

        return centroid

    def _update_centroids(self):
        for index, samples in self.centroid_sample_mapping.items():
            self.centroids[index] = np.mean(samples, axis=0)

    def _get_minimal_distance_centroid(self, sample):
        centroids_distance_from_record = list(map(lambda x: distance.euclidean(x, sample), self.centroids))
        centroid = np.argmin(centroids_distance_from_record)
        return centroid

    def _assign_samples_to_centroids(self):
        centroid_sample_mapping = dict()
        sample_centroid_mapping = dict()

        for index in range(self.k):
            centroid_sample_mapping[index] = list()

        for sample_index, sample_features in self.data.iterrows():
            centroid = self._get_minimal_distance_centroid(sample_features)
            centroid_sample_mapping[centroid].append(sample_features)
            sample_centroid_mapping[sample_index] = centroid

        return centroid_sample_mapping, sample_centroid_mapping

    def _calc_score(self):
        score = 0
        for centroid, samples in self.centroid_sample_mapping.items():
            score += sum(list(map(lambda x: distance.euclidean(x, self.centroids[centroid]), samples)))
        return score

    def fit(self, data, iterations):
        self.data = data
        self._initilize_centroids()

        last_score = None

        for i in range(iterations):
            self.centroid_sample_mapping, self.sample_centroid_mapping = self._assign_samples_to_centroids()
            self._update_centroids()

            if last_score:
                score = self._calc_score()
                if (last_score-score)<0.001:
                    last_score = score
                    break
                last_score = score

            else:
                last_score = self._calc_score()

        self.score = last_score
        return self

    def plot_elbow(self,y):
        plt.plot(1+np.arange(len(y)), y)
        plt.ylabel('some numbers')
        plt.show()

    @property
    def get_centorids(self):
        return self.centroids

    @property
    def get_score(self):
        return self.score

    @property
    def get_centroid_mapping(self):
        return self.centroid_sample_mapping

    @property
    def get_samples_mapping(self):
        return self.sample_centroid_mapping