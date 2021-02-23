"""
Code to find correspondances between two point clouds
"""
import torch
from pytorch3d.ops.knn import knn_points


@torch.jit.script
def calculate_ratio_test(dists):
    """
    Calculate weights for matches based on the ratio between kNN distances.

    Input:
        (N, P, 2) Cosine Distance between point and nearest 2 neighbors
    Output:
        (N, P, 1) Weight based on ratio; higher is more unique match
    """
    # Convert points so that 0 means perfect similarity and clamp to avoid numerical
    # instability
    dists = (1 - dists).clamp(min=1e-9)

    # Ratio -- close to 0 is completely unique; 1 is same feature
    # Weight -- Convert so that higher is more unique
    ratio = dists[:, :, 0:1] / dists[:, :, 1:2]
    """
    I invert the weight to have higher is better, but I still don't have a very good
    intuition about what's actually being represented here. The ratio is dimensionless,
    so it's scale invariant, but it's not invariant to the distribution of features over
    the whole space. eg, if the network uses a small subspace to represent things, then
    the ratios will, on average, be lower than if it was distributed over the whole
    space, assuming that the correct correspondances feature pair only differs by a
    noise that doesn't depend on the distribution ... and who knows if that's true?

    It might make sense to consider other weighting schemes; eg, learned, or even
    ratio_test with some learned parameter. For now, I will use the simplest version of
    the ratio test; no thresholding, or reweighting.
    """
    weight = 1 - ratio

    return weight


@torch.jit.script
def get_topk_matches(dists, idx, num_corres: int):
    dist, idx_source = torch.topk(dists, k=num_corres, dim=1)
    idx_target = idx.gather(1, idx_source)
    return idx_source, idx_target, dist


def get_correspondences(
    P1, P2, num_corres, P1_X, P2_X, metric="cosine", ratio_test=False,
):
    """
    Finds the kNN according to either euclidean distance or cosine distance. This is
    tricky since PyTorch3D's fast kNN kernel does euclidean distance, however, we can
    take advantage of the relation between euclidean distance and cosine distance for
    points sampled on an n-dimension sphere.

    Using the quadratic expansion, we find that finding the kNN between two normalized
    is the same regardless of whether the metric is euclidean distance or cosine
    similiarity.

        -2 * xTy = (x - y)^2 - x^2 - y^2
        -2 * xtY = (x - y)^2 - 1 - 1
        - xTy = 0.5 * (x - y)^2 - 1

    Hence, the metric that would maximize cosine similarity is the same as that which
    would minimize the euclidean distance between the points, with the distances being
    a simple linear transformation.

    Input:
        P1          FloatTensor (N x P x F)     features for first pointcloud
        P2          FloatTensor (N x Q x F)     features for second pointcloud
        num_corres  Int                         number of correspondances
        P1_X        FloatTensor (N x P x 3)     xyz for first pointcloud
        P2_X        FloatTensor (N x Q x 3)     xyz for second pointcloud
        metric      {cosine, euclidean}         metric to be used for kNN
        ratio_test  Boolean                     whether to use ratio test for kNN

    Returns:
        LongTensor (N x 2 * num_corres)         Indices for first pointcloud
        LongTensor (N x 2 * num_corres)         Indices for second pointcloud
        FloatTensor (N x 2 * num_corres)        Weights for each correspondace
        FloatTensor (N x 2 * num_corres)        Cosine distance between features
    """
    batch_size, num_points, feature_dimension = P1.shape
    assert metric in ["euclidean", "cosine"]

    if metric == "cosine":
        # Normalize points -- clamp to deal with missing points for less dense models.
        # Those points will have a cosine weight of 0.5, but will get filtered out below
        # as invalid points.
        P1 = P1 / P1.norm(dim=2, keepdim=True).clamp(min=1e-9)
        P2 = P2 / P2.norm(dim=2, keepdim=True).clamp(min=1e-9)

    if ratio_test:
        K = 2
    else:
        K = 1

    # Calculate kNN for k=2; both outputs are (N, P, K)
    # idx_1 returns the indices of the nearest neighbor in P2
    dists_1, idx_1, _ = knn_points(P1, P2, K=K)
    dists_2, idx_2, _ = knn_points(P2, P1, K=K)

    # Take the nearest neighbor for the indices for k={1, 2}
    idx_1 = idx_1[:, :, 0:1]
    idx_2 = idx_2[:, :, 0:1]

    # Transform euclidean distance of points on a sphere to cosine similarity
    cosine_1 = 1 - 0.5 * dists_1
    cosine_2 = 1 - 0.5 * dists_2

    if metric == "cosine":
        dists_1 = cosine_1
        dists_2 = cosine_2

    # Apply ratio test
    if ratio_test:
        weights_1 = calculate_ratio_test(dists_1)
        weights_2 = calculate_ratio_test(dists_2)
    else:
        weights_1 = dists_1[:, :, 0:1]
        weights_2 = dists_2[:, :, 0:1]

    # find if both the points in the correspondace are valid
    valid_z1 = P1_X[:, :, 2] != 0
    valid_z2 = P2_X[:, :, 2] != 0
    valid_znn1 = P2_X[:, :, 2].gather(1, idx_1.squeeze(2)) != 0.0
    valid_znn2 = P1_X[:, :, 2].gather(1, idx_2.squeeze(2)) != 0.0
    valid_1 = (valid_z1 & valid_znn1).float()
    valid_2 = (valid_z2 & valid_znn2).float()

    # multiple by valid pixels
    weights_1 = weights_1 * valid_1.unsqueeze(2)
    weights_2 = weights_2 * valid_2.unsqueeze(2)

    # Get topK matches in both directions
    m12_idx1, m12_idx2, m12_dist = get_topk_matches(weights_1, idx_1, num_corres)
    m21_idx2, m21_idx1, m21_dist = get_topk_matches(weights_2, idx_2, num_corres)
    cosine_1 = cosine_1[:, :, 0:1].gather(1, m12_idx1)
    cosine_2 = cosine_2[:, :, 0:1].gather(1, m21_idx2)

    # concatenate into correspondances and weights
    matches_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1).squeeze(dim=2)
    matches_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1).squeeze(dim=2)
    matches_dist = torch.cat((m12_dist, m21_dist), dim=1).squeeze(dim=2)
    matches_cosn = torch.cat((cosine_1, cosine_2), dim=1).squeeze(dim=2)

    return matches_idx1, matches_idx2, matches_dist, matches_cosn
