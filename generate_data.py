import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set random seed for reproducibility
np.random.seed(42)

save_dir = '/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-2/'


def generate_3d_data():
    # Three random clusters in 3D space
    mean_1 = [1, 1, 1]
    mean_2 = [5, 5, 5]
    mean_3 = [9, 9, 9]

    cov = [[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]]  # Diagonal covariance matrix

    # Generate 3 clusters
    cluster_1 = np.random.multivariate_normal(mean_1, cov, 100)
    cluster_2 = np.random.multivariate_normal(mean_2, cov, 100)
    cluster_3 = np.random.multivariate_normal(mean_3, cov, 100)

    # Stack the clusters together and assign labels for each cluster
    data = np.vstack((cluster_1, cluster_2, cluster_3))
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)  # Labels for each cluster

    return data, labels


def save_data_as_csv(data, labels):
    # Combine data and labels
    df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
    df['Cluster'] = labels
    csv_path = os.path.join(save_dir, '3d_data.csv')

    # Save dataset to CSV
    df.to_csv(csv_path, index=False)
    print(f'Data saved as CSV at {csv_path}')


def visualize_data(data, labels):
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color assignment based on clusters
    colors = ['red', 'green', 'blue']

    for cluster in np.unique(labels):
        ax.scatter(data[labels == cluster, 0], data[labels == cluster, 1], data[labels == cluster, 2],
                   color=colors[cluster], label=f'Cluster {cluster}')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.title('Generated 3D Data with 3 Clusters')
    plt.legend()

    # Save plot as PNG
    png_path = os.path.join(save_dir, 'initial_data.png')
    plt.savefig(png_path)
    plt.show()
    print(f'Plot saved as PNG at {png_path}')


if __name__ == "__main__":
    data, labels = generate_3d_data()
    save_data_as_csv(data, labels)
    visualize_data(data, labels)



