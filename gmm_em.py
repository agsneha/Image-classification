import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
from generate_data import generate_3d_data

save_dir = '/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-2/'


class GMM_EM:
    def __init__(self, X, n_clusters):
        self.X = X
        self.n_clusters = n_clusters
        self.n_samples, self.n_features = X.shape

        # Initialize parameters randomly
        self.means = np.random.rand(n_clusters, self.n_features)
        self.covs = np.array([np.eye(self.n_features) for _ in range(n_clusters)])
        self.weights = np.ones(n_clusters) / n_clusters
        self.responsibilities = np.zeros((self.n_samples, n_clusters))

    def e_step(self):
        # Expectation step: Compute responsibilities
        for k in range(self.n_clusters):
            self.responsibilities[:, k] = self.weights[k] * multivariate_normal(self.means[k], self.covs[k]).pdf(self.X)
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)

    def m_step(self):
        # Maximization step: Update parameters
        Nk = self.responsibilities.sum(axis=0)

        # Update means
        for k in range(self.n_clusters):
            self.means[k] = np.sum(self.responsibilities[:, k][:, np.newaxis] * self.X, axis=0) / Nk[k]

            # Update covariance
            diff = self.X - self.means[k]
            self.covs[k] = np.dot(self.responsibilities[:, k] * diff.T, diff) / Nk[k]

        # Update weights
        self.weights = Nk / self.n_samples

    def log_likelihood(self):
        # Compute log likelihood for convergence
        log_likelihood = 0
        for k in range(self.n_clusters):
            log_likelihood += np.sum(
                self.responsibilities[:, k] * np.log(multivariate_normal(self.means[k], self.covs[k]).pdf(self.X)))
        return log_likelihood

    def train(self, n_iters=100):
        images = []

        # Step 0: Visualize the initial dataset
        self.visualize_initial_dataset(images)

        # Iterative EM steps for n_iters iterations
        for i in range(n_iters):
            self.e_step()  # Perform expectation step
            self.m_step()  # Perform maximization step
            ll = self.log_likelihood()
            print(f"Iteration {i + 1}, Log-Likelihood: {ll}")

            # Save intermediate steps (for every 10 steps as an example)
            if (i + 1) % 10 == 0:
                self.visualize_gmm(i + 1, images)

        # Save final plot
        self.visualize_gmm('final', images)

        # Create video of the entire process
        video_path = os.path.join(save_dir, 'gmm_full_process.mp4')
        self.create_video(images, video_path)
        print(f'Video saved at {video_path}')

    def visualize_initial_dataset(self, images):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], color='gray', label='Initial Data')
        ax.set_title('Initial 3D Dataset')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.legend()

        # Save the initial dataset visualization as an image
        img_path = os.path.join(save_dir, 'gmm_initial_data.png')
        plt.savefig(img_path)
        images.append(img_path)
        plt.close()

    def visualize_gmm(self, step, images):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Assign cluster colors
        colors = ['red', 'green', 'blue']

        # Plot clusters based on responsibilities
        for k in range(self.n_clusters):
            cluster_points = self.X[self.responsibilities[:, k] > 0.5]  # Points with high responsibility
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                       color=colors[k], label=f'Cluster {k}')

        ax.set_title(f'GMM Clustering after {step} EM Steps' if step != 'final' else 'Final GMM Clustering')
        plt.legend()

        # Save each step as an image
        img_path = os.path.join(save_dir, f'gmm_step_{step}.png')
        plt.savefig(img_path)
        images.append(img_path)
        plt.close()

    def create_video(self, images, video_path):
        # Create video from saved images
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

        for image in images:
            video.write(cv2.imread(image))

        cv2.destroyAllWindows()
        video.release()


if __name__ == "__main__":
    # Generate data and apply GMM
    data, labels = generate_3d_data()
    gmm = GMM_EM(data, n_clusters=3)
    gmm.train(n_iters=100)  # Running 100 iterations of GMM
