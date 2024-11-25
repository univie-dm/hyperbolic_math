import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.spatial import distance_matrix
from tqdm import tqdm


def get_delta(loader, device='cpu'):
    """
    Computes delta value for image data by extracting features using ResNet34.
    """
    resnet34 = torchvision.models.resnet34(pretrained=True)
    # Remove the final fully connected layer
    resnet34_features = nn.Sequential(*list(resnet34.children())[:-1])
    resnet34_features = resnet34_features.to(device)
    resnet34_features.eval()

    all_features = []
    for i, (batch, _) in enumerate(tqdm(loader, desc="Processing batches")):
        with torch.no_grad():
            batch = batch.to(device)
            features = resnet34_features(batch).squeeze()
            all_features.append(features.detach().cpu().numpy())

    all_features = np.concatenate(all_features)
    idx = np.random.choice(len(all_features), min(1500, len(all_features)))
    all_features_small = all_features[idx]

    dists = distance_matrix(all_features_small, all_features_small)
    delta = delta_hyp(dists)
    diam = np.max(dists)
    return delta, diam, all_features

def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)

def batched_delta_hyp(X, n_tries=10, batch_size=1500):
    vals = []
    for _ in tqdm(range(n_tries), desc="Batched delta computation"):
        idx = np.random.choice(len(X), batch_size)
        X_batch = X[idx]
        distmat = distance_matrix(X_batch, X_batch)
        diam = np.max(distmat)
        delta_rel = 2 * delta_hyp(distmat) / diam
        vals.append(delta_rel)
    return np.mean(vals), np.std(vals)
