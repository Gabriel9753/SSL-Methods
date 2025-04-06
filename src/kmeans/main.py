import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns
import matplotlib.pyplot as plt
from dataset_downloader.dataset_utils import get_dataset_dfs, load_images
from sklearn.metrics import accuracy_score, confusion_matrix
from rich import print
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CUR_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def infer_cluster_labels(kmeans, actual_labels):
    inferred_labels = defaultdict(list)

    for i in range(kmeans.n_clusters):
        labels = []
        index = np.where(kmeans.labels_ == i)

        labels.append(actual_labels[index])

        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        inferred_labels[np.argmax(counts)].append(i)

    return inferred_labels


def infer_data_labels(X_labels, cluster_labels):
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels


def create_cluster_grid(cluster_centers, grid_size=None):
    n_clusters = cluster_centers.shape[0]
    img_height, img_width = cluster_centers.shape[1], cluster_centers.shape[2]

    if grid_size is None:
        grid_cols = int(np.ceil(np.sqrt(n_clusters)))
        grid_rows = int(np.ceil(n_clusters / grid_cols))
    else:
        grid_rows, grid_cols = grid_size

    grid_image = np.ones((grid_rows * img_height, grid_cols * img_width)) * 255

    for i in range(n_clusters):
        row = i // grid_cols
        col = i % grid_cols

        y_start = row * img_height
        y_end = (row + 1) * img_height
        x_start = col * img_width
        x_end = (col + 1) * img_width

        grid_image[y_start:y_end, x_start:x_end] = cluster_centers[i]

    return grid_image


def apply_pca(train_vectors, test_vectors, n_components=50):
    """Apply PCA dimensionality reduction"""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vectors)
    test_scaled = scaler.transform(test_vectors)

    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    explained_variance = sum(pca.explained_variance_ratio_)
    print(f"PCA with {n_components} components explains {explained_variance:.4f} of variance")

    return train_pca, test_pca


def apply_tsne(train_vectors, test_vectors, n_components=2, perplexity=5.0):
    """Apply t-SNE dimensionality reduction"""
    train_pca, test_pca = apply_pca(train_vectors, test_vectors, n_components=50)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    train_tsne = tsne.fit_transform(train_pca)

    tsne_test = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    test_tsne = tsne_test.fit_transform(test_pca)

    return train_tsne, test_tsne


def visualize_clusters(features, labels, predicted_labels, title, filename):
    """Visualize clusters in 2D"""
    plt.figure(figsize=(12, 10))

    max_points = 5000
    if len(features) > max_points:
        indices = np.random.choice(len(features), max_points, replace=False)
        features_sample = features[indices]
        labels_sample = labels[indices]
        predicted_labels_sample = predicted_labels[indices]
    else:
        features_sample = features
        labels_sample = labels
        predicted_labels_sample = predicted_labels

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_sample[:, 0], features_sample[:, 1], c=labels_sample, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f"{title} - Ground Truth")

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(
        features_sample[:, 0], features_sample[:, 1], c=predicted_labels_sample, cmap="tab10", s=5, alpha=0.7
    )
    plt.colorbar(scatter)
    plt.title(f"{title} - KMeans Clusters")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close()


def run_kmeans(train_vectors, test_vectors, train_labels, test_labels, n_clusters, description="Raw"):
    """Run KMeans clustering and evaluate results"""
    print(f"\n--- Running KMeans with {description} data ---")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, max_iter=300)
    kmeans.fit(train_vectors)
    train_cluster_labels = kmeans.labels_
    test_cluster_labels = kmeans.predict(test_vectors)

    cluster_labels = infer_cluster_labels(kmeans, train_labels)
    predicted_labels_train = infer_data_labels(train_cluster_labels, cluster_labels)
    predicted_labels_test = infer_data_labels(test_cluster_labels, cluster_labels)

    train_accuracy = accuracy_score(train_labels, predicted_labels_train)
    test_accuracy = accuracy_score(test_labels, predicted_labels_test)
    print(f"{description} Train Accuracy: {train_accuracy:.4f}")
    print(f"{description} Test Accuracy: {test_accuracy:.4f}")

    if train_vectors.shape[1] == 2:
        visualize_clusters(
            train_vectors,
            train_labels,
            predicted_labels_train,
            f"KMeans with {description}",
            os.path.join(RESULTS_DIR, f"kmeans_{description.lower()}_visualization.png"),
        )

    return kmeans, train_accuracy, test_accuracy, predicted_labels_train, predicted_labels_test, cluster_labels


def main():
    # load dataframes
    dfs = get_dataset_dfs("data/mnist")
    train_df = dfs["train"]
    train_df["class_name"] = train_df["class_name"].cat.codes
    train_df["class_name"] = train_df["class_name"].astype("int")
    test_df = dfs["test"]
    test_df["class_name"] = test_df["class_name"].cat.codes
    test_df["class_name"] = test_df["class_name"].astype("int")

    # load images for train and test splits
    train_images = load_images(train_df["image_path"].tolist(), max_workers=8, mode="gray", target_size=(28, 28))
    test_images = load_images(test_df["image_path"].tolist(), max_workers=8, mode="gray", target_size=(28, 28))

    train_vectors = np.array([img.flatten() for img in train_images])
    test_vectors = np.array([img.flatten() for img in test_images])

    train_vectors = train_vectors.astype("float32") / 255.0
    test_vectors = test_vectors.astype("float32") / 255.0

    # Original KMeans with raw pixels
    n_clusters = len(train_df["class_name"].unique())
    kmeans, train_accuracy, test_accuracy, predicted_labels, predicted_labels_test, cluster_labels = run_kmeans(
        train_vectors,
        test_vectors,
        train_df["class_name"].values,
        test_df["class_name"].values,
        n_clusters,
        "Raw Pixels",
    )

    # KMeans with PCA
    n_components_pca = min(2, train_vectors.shape[1])  # Choose appropriate number of components
    train_pca, test_pca = apply_pca(train_vectors, test_vectors, n_components=n_components_pca)
    run_kmeans(train_pca, test_pca, train_df["class_name"].values, test_df["class_name"].values, n_clusters, "PCA")

    # KMeans with t-SNE (2D for visualization)
    # train_tsne, test_tsne = apply_tsne(train_vectors, test_vectors, n_components=2)
    # run_kmeans(train_tsne, test_tsne, train_df["class_name"].values, test_df["class_name"].values, n_clusters, "t-SNE")

    cluster_centers = kmeans.cluster_centers_
    centers_as_images = cluster_centers.reshape(-1, 28, 28) * 255
    centers_as_images = centers_as_images.astype(np.uint8)

    reordered_centers = []
    for digit in range(10):
        for cluster_idx in cluster_labels[digit]:
            reordered_centers.append((digit, cluster_idx, centers_as_images[cluster_idx]))
    reordered_centers.sort(key=lambda x: x[0])

    ordered_centers = np.array([center[2] for center in reordered_centers])
    grid_image = create_cluster_grid(ordered_centers)

    plt.figure(figsize=(12, 8))
    plt.imshow(grid_image, cmap="gray")
    plt.title("KMeans Cluster Centers (MNIST)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cluster_centers_visualization.png"), dpi=300)
    print(f"Saved visualization to {os.path.join(RESULTS_DIR, 'cluster_centers_visualization.png')}")

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(test_df["class_name"].values, predicted_labels_test)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (KMeans on Raw Pixels)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    print(f"Saved confusion matrix to {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()
