import os
import random
import re
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, pairwise_distances)
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import hyperparams

# ------------------- GPU & TF INFO -------------------
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus if gpus else "No GPU found")

tf.debugging.set_log_device_placement(False)
# -----------------------------------------------------

# ------------------- CONFIG -------------------
DATASET_SIZES = [456, 228, 100, 50]
CLUSTERS = 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
OUTPUT_CSV_FILENAME = './ABIDE - Cross Validation Data/random_cluster_experiment_results'
BASE_DIRS = {
    'benign': './ABIDE imaging data/benign/',
    'malignant': './ABIDE imaging data/malignant/'
}
RANDOM_SEED = 42
# ----------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Get all image paths
def get_image_paths():
    paths = {
        'benign': sorted(glob(os.path.join(BASE_DIRS['benign'], '*'))),
        'malignant': sorted(glob(os.path.join(BASE_DIRS['malignant'], '*')))
    }
    return paths

# Sample balanced dataset safely
def sample_dataset(image_paths, total_size):
    max_per_class = min(len(image_paths['benign']), len(image_paths['malignant']))
    per_class = min(total_size // 2, max_per_class)

    print(f"Sampling {per_class} images per class (benign/malignant).")

    return (
        random.sample(image_paths['benign'], per_class) +
        random.sample(image_paths['malignant'], per_class)
    )

def get_dataset_sizes():
    benign_count = len(glob(os.path.join(BASE_DIRS['benign'], '*')))
    malignant_count = len(glob(os.path.join(BASE_DIRS['malignant'], '*')))
    max_size = min(benign_count, malignant_count) * 2

    return [s for s in [456, 228, 100, 50] if s <= max_size]

# Extract patient ID from filename
def extract_patient_id(filename):
    match = re.match(r".*(patient_\d+)", os.path.basename(filename))
    return match.group(1) if match else os.path.splitext(os.path.basename(filename))[0]

# Load and preprocess images
def load_images(paths):
    images, labels, patients = [], [], []
    for path in paths:
        img = load_img(path, target_size=IMG_SIZE)
        arr = preprocess_input(img_to_array(img))
        label = 0 if 'benign' in path else 1
        images.append(arr)
        labels.append(label)
        patients.append(extract_patient_id(path))
    return np.array(images), np.array(labels), np.array(patients)

# Randomly assign cluster labels and dummy distance metrics
def cluster_features(n_samples, n_clusters=4, feature_dim=128):
    cluster_labels = np.random.randint(0, n_clusters, size=n_samples)

    # Generate random dummy centroids for inter-cluster distance
    dummy_centroids = np.random.rand(n_clusters, feature_dim)
    inter_cluster_matrix = pairwise_distances(dummy_centroids)

    intra_dists = []
    for i in range(n_clusters):
        members = np.random.rand(np.sum(cluster_labels == i), feature_dim)
        if len(members) > 1:
            intra_dists.append(pairwise_distances(members).mean())
        else:
            intra_dists.append(0)

    return cluster_labels, inter_cluster_matrix, intra_dists

# Build classifier
def build_classifier():
    activation, optimizer, loss, metrics = (
        hyperparams.activation, hyperparams.optimizer,
        hyperparams.loss, hyperparams.metrics
    )

    eff = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    model = Sequential([
        eff,
        Flatten(),
        Dense(1, activation=activation)
    ])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

# Evaluate using grouped cross-validation and return per-sample info
def evaluate(images, labels, groups):
    kf = GroupKFold(n_splits=4)
    y_true_all, y_pred_all, y_prob_all, group_all = [], [], [], []

    for train_idx, test_idx in kf.split(images, labels, groups):
        model = build_classifier()
        model.fit(images[train_idx], labels[train_idx], epochs=5, batch_size=32, verbose=0)
        probs = model.predict(images[test_idx]).flatten()
        preds = (probs > 0.5).astype(int)

        y_true_all.extend(labels[test_idx])
        y_pred_all.extend(preds)
        y_prob_all.extend(probs)
        group_all.extend(groups[test_idx])

    metrics = {
        'accuracy': accuracy_score(y_true_all, y_pred_all),
        'precision': precision_score(y_true_all, y_pred_all),
        'recall': recall_score(y_true_all, y_pred_all),
        'f1': f1_score(y_true_all, y_pred_all),
        'auc': roc_auc_score(y_true_all, y_prob_all)
    }

    return metrics, np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)

# Main experiment loop
def run_experiment(iteration):
    all_paths = get_image_paths()
    print(f"Available images - Benign: {len(all_paths['benign'])}, Malignant: {len(all_paths['malignant'])}")
    results = []

    for size in get_dataset_sizes():
        try:
            print(f"\nRunning experiment with dataset size: {size}")
            paths = sample_dataset(all_paths, size)
            images, labels, patients = load_images(paths)

            print("Assigning random clusters...")
            cluster_labels, inter_matrix, intra_dists = cluster_features(len(images))

            print("Evaluating model...")
            metrics, y_true, y_pred, y_prob = evaluate(images, labels, patients)

            for cluster_id in range(CLUSTERS):
                cluster_mask = (cluster_labels == cluster_id)

                if np.sum(cluster_mask) == 0:
                    print(f"Skipping empty cluster {cluster_id}")
                    continue

                true_cluster = y_true[cluster_mask]
                pred_cluster = y_pred[cluster_mask]
                prob_cluster = y_prob[cluster_mask]

                try:
                    row = {
                        'dataset_size': size,
                        'cluster_label': cluster_id,
                        'cluster_size': int(np.sum(cluster_mask)),
                        'accuracy': accuracy_score(true_cluster, pred_cluster),
                        'precision': precision_score(true_cluster, pred_cluster),
                        'recall': recall_score(true_cluster, pred_cluster),
                        'f1': f1_score(true_cluster, pred_cluster),
                        'auc': roc_auc_score(true_cluster, prob_cluster),
                        'intra_cluster_dist': intra_dists[cluster_id]
                    }
                except ValueError:
                    # In case only one class is present in this cluster
                    row = {
                        'dataset_size': size,
                        'cluster_label': cluster_id,
                        'cluster_size': int(np.sum(cluster_mask)),
                        'accuracy': None,
                        'precision': None,
                        'recall': None,
                        'f1': None,
                        'auc': None,
                        'intra_cluster_dist': intra_dists[cluster_id]
                    }

                # Add inter-cluster distances
                for i in range(CLUSTERS):
                    for j in range(i + 1, CLUSTERS):
                        row[f'inter_cluster_{i}_{j}'] = inter_matrix[i, j]

                results.append(row)

            print(f"✅ Dataset {size} done.")

        except ValueError as ve:
            print(f"⚠️ Skipping dataset size {size}: {ve}")
        except Exception as e:
            print(f"❌ Error during experiment size {size}: {e}")

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"{OUTPUT_CSV_FILENAME}_{iteration}.csv", index=False)
        print(f"\n🎉 All results written to: {OUTPUT_CSV_FILENAME}_{iteration}.csv")
    else:
        print("⚠️ No results to write.")

if __name__ == "__main__":
    for i in range(5):
        run_experiment(i)