from distance import Distance
from operations import Operation
import matplotlib.pyplot as plt
import random

op = Operation()


class KmeansFC:
    def __init__(self, k=5, max_iters=100, plot_process=False) -> None:
        self.k = k
        self.max_iters = max_iters
        self.plot_process = plot_process
        self.clusters = None
        self.centroids = None

    def fit_predict(self, X):
        self.X_train = X
        self.n_samples, self.n_features = len(X), len(X[0])

        # init centroids (I use random data points in training set as init centroids)
        init_centroids_id = random.sample(range(0, self.n_samples), self.k)
        self.centroids = [self.X_train[i] for i in init_centroids_id]

        # Converge Process
        for _ in range(self.max_iters):
            # update cluster
            self.clusters = self._clasify_clusters()

            # plot
            if self.plot_process:
                self.plot()

            # update centroids
            old_centroids = self.centroids
            self.centroids = self._new_centroids()

            # check if converger
            if self._is_converged(old_centroids, self.centroids):
                break

            # plot
            if self.plot_process:
                self.plot()

    def _clasify_clusters(self):
        '''
        Assign data points to nearest centroid and make cluster
        '''
        clusters = [[] for _ in range(self.k)]
        for i, point in enumerate(self.X_train):
            centroid_id = self._nearest_centroid(point)
            clusters[centroid_id].append(i)
        return clusters

    def _nearest_centroid(self, data_point):
        distances = [Distance.euclidean(data_point, c) for c in self.centroids]
        nearest_centroid_id = op.argmin(distances)
        return nearest_centroid_id

    def _new_centroids(self):
        new_centroids = [[0] * self.n_features] * self.k
        for cluster_id, cluster_list in enumerate(self.clusters):
            cluster_coor_list = [self.X_train[c] for c in cluster_list]
            cluster_mean = op.mean(cluster_coor_list, axis=0)
            new_centroids[cluster_id] = cluster_mean
        return new_centroids

    def _is_converged(self, old_centroids, current_centroids):
        distances = [Distance.euclidean(old_centroids[i], current_centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def plot(self):
        _, ax = plt.subplots(figsize=(12, 8))

        for _, idx_list in enumerate(self.clusters):
            coor_list = [self.X_train[i] for i in idx_list]
            point = op.transpose(coor_list)
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidths=1)

        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=3)

    clusters = len(set(y))

    k = KmeansFC(k=clusters, max_iters=150, plot_process=True)
    y_pred = k.fit_predict(X.tolist())
    # k.plot()
