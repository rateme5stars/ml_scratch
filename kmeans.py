import numpy as np
from distance import Distance
import matplotlib.pyplot as plt


class KmeansNumpy:
    def __init__(self, k=5, max_iters=100, plot_process=False) -> None:
        self.k = k
        self.max_iters = max_iters
        self.plot_process = plot_process

        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for i, sample in enumerate(self.X):
            centroid_id = self._nearest_centroid(sample, centroids)
            clusters[centroid_id].append(i)
        return clusters

    def _nearest_centroid(self, sample, centroids):
        distances = [Distance.euclidean(sample, c) for c in centroids]
        nearest_id = np.argmin(distances)
        return nearest_id

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_id, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_id] = cluster_mean
        return centroids

    def _is_converged(self, old_centroid, centroids):
        distances = [Distance.euclidean(old_centroid[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def _get_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_id, cluster in enumerate(clusters):
            for sample_id in cluster:
                labels[sample_id] = cluster_id
        return labels

    def plot(self):
        _, ax = plt.subplots(figsize=(12, 8))

        for _, idx in enumerate(self.clusters):
            point = self.X[idx].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidths=1)

        plt.show()

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # init centroids
        init_centroid_coor = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[i] for i in init_centroid_coor]

        # fitting
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_process:
                self.plot()
                # update centroids
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # check if converged
            if self._is_converged(old_centroids, self.centroids):
                break
        # return cluster labels
        return self._get_labels(self.clusters)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y = make_blobs(centers=2, n_samples=500, n_features=2, shuffle=True, random_state=10)

    clusters = len(np.unique(y))

    k = KmeansNumpy(k=clusters, max_iters=150, plot_process=False)
    y_pred = k.predict(X)
    k.plot()
