from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm


def dist(a, b, axis):
    return np.linalg.norm(a - b, axis=axis)


def run_kmeans(data_points, K, n_iter=100):
    num_data_points = data_points.shape[0]

    # Step 1 : Pick K random points
    center_indices = np.random.randint(num_data_points, size=K)
    center_points = data_points[center_indices]

    for _ in tqdm(range(n_iter), ncols=70, desc="K={}".format(K)):
        clusters = {i: [] for i in range(K)}
        errors = 0.0
        # Step 2 : Assign each data points to cluster
        for i in range(num_data_points):
            distances = dist(data_points[i], center_points, axis=1)
            cluster, error = np.argmin(distances), np.min(distances)
            clusters[cluster].append(i)
            errors += error

        # Step 3 : Find new centroids
        for i in range(K):
            center_points[i] = np.mean(data_points[clusters[i]], axis=0)

    # Print results
    for i in range(K):
        print("Cluster number {}".format(i))
        print(data_points[clusters[i]])

    print("MSE (num cluster {}) : {}".format(K, errors))


def main():
    data_points = []
    with open('prep_data.csv', 'r') as fp:
        for idx, line in enumerate(fp):
            if idx == 0:
                continue

            data_point = np.fromstring(line, dtype=float, sep='\t')
            data_points.append(data_point)
    data_points = np.array(data_points)[:, 1:]

    run_kmeans(data_points, 2)


if __name__ == '__main__':
    main()
