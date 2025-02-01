import numpy as np
import cv2


def get_dominant_colors(pixels, k=5, tolerance=50, black_threshold=100):
    """Gets the k dominant colors in an image using K-Means clustering."""

    # Reshape the image into a 2D array of pixels
    pixels = np.reshape(pixels, (-1, 3))
    pixels = np.float32(pixels)

    # Perform K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Convert the centers back to uint8
    centers = np.uint8(centers)

    # Count the number of pixels in each cluster
    _, counts = np.unique(labels, return_counts=True)

    # Sort the colors by their frequency
    indices = np.argsort(counts)[::-1]
    dominant_colors = centers[indices]

    # remove colors too close to black
    dominant_colors = dominant_colors[
        np.max(dominant_colors, axis=-1) > black_threshold
    ]

    # remove colors too close to each other
    chosen_dominant_colors = []
    for color in dominant_colors:
        if len(chosen_dominant_colors) == 0:
            chosen_dominant_colors.append(color)
            continue

        far_enough = True
        for color_chosen in chosen_dominant_colors:
            distance = np.max(color - color_chosen)
            if distance < tolerance:
                far_enough = False
                break

        if far_enough:
            chosen_dominant_colors.append(color)

    return chosen_dominant_colors


def remove_noise(input_array, min_size=15, threshold=10):
    in_mask = np.any(input_array > threshold, axis=-1).astype(np.uint8) * 255
    _, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
        in_mask, connectivity=8
    )

    sizes = stats[:, cv2.CC_STAT_AREA][..., None]

    im_result = np.where(
        sizes[im_with_separated_blobs] >= min_size, input_array, 0
    )
    return im_result


def extract_features(image_array_segment, tolerance=50, overlap_threshold=0.9):
    image_array_segment = np.asarray(image_array_segment)
    tolerance = np.array([tolerance, tolerance, tolerance])
    features = []
    masks = []
    wing_mask = np.any(image_array_segment > 0, axis=-1)
    wing_mask = np.tile(wing_mask[..., None], (1, 1, 3)).astype(np.uint8) * 8
    for target_color in get_dominant_colors(image_array_segment, k=8):
        if np.max(target_color) < 100:
            continue
        lower_bound = target_color - tolerance
        upper_bound = target_color + tolerance

        mask = np.logical_and(
            image_array_segment > lower_bound,
            image_array_segment < upper_bound,
        )
        mask = np.all(mask, axis=-1)

        result = np.copy(wing_mask)
        result[mask] = image_array_segment[mask]

        # clean out noise
        result = remove_noise(result, min_size=15)

        # remove image if not enough pixels
        if np.sum(result > 10) < 100:
            continue

        # check if overlap with existing features is too large
        too_much_overlap = False
        for idx, mask_f in enumerate(masks):
            overlap = np.sum(np.logical_and(mask, mask_f)) / np.sum(mask_f)
            if overlap > overlap_threshold:
                too_much_overlap = True
                break
        if too_much_overlap:
            # take larger mask
            if np.sum(mask) > np.sum(mask_f):
                features[idx] = result
                masks[idx] = mask
            continue

        features.append(result)
        masks.append(mask)
    return features


def get_major_axis_orientation(image, threshold=1):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return 0
    contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(contour)
    return ellipse, cv2.contourArea(contour)


def rotate_image(image):
    ellipse, size = get_major_axis_orientation(image)
    angle = ellipse[2]
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR
    )
    return result, size


def compute_overlap(image1, image2, threshold=1):
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    _, image1 = cv2.threshold(image1, threshold, 255, cv2.THRESH_BINARY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    _, image2 = cv2.threshold(image2, threshold, 255, cv2.THRESH_BINARY)
    overlap = cv2.bitwise_and(image1, image2)
    return np.sum(overlap) / np.sum(image1)


def flip_image(image):
    return cv2.flip(image, 1)


def align_image(image1, image2):
    ellipse1, _ = get_major_axis_orientation(image1)
    ellipse2, _ = get_major_axis_orientation(image2)
    angle1 = ellipse1[2]
    angle2 = ellipse2[2]
    angle_diff = angle2 - angle1

    # get center from ellipse
    center1 = (int(ellipse1[0][0]), int(ellipse1[0][1]))
    center2 = (int(ellipse2[0][0]), int(ellipse2[0][1]))

    # translate image2 to center of image1
    translation = (center1[0] - center2[0], center1[1] - center2[1])
    image2 = cv2.warpAffine(
        image2,
        np.float32([[1, 0, translation[0]], [0, 1, translation[1]]]),
        image2.shape[:2],
    )

    center = (image1.shape[1] // 2, image1.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_diff, 1.0)
    result = cv2.warpAffine(
        image2, rot_mat, image2.shape[:2], flags=cv2.INTER_LINEAR
    )
    return result


def align_images(image1, image2):
    image2_tr = align_image(image1, image2)
    image2_tr_flip = align_image(image1, flip_image(image2))

    overlap1 = compute_overlap(image1, image2_tr)
    overlap2 = compute_overlap(image1, image2_tr_flip)

    if overlap1 > overlap2:
        image2_final = image2_tr
    else:
        image2_final = image2_tr_flip

    return image1, image2_final


def overlay_images(image1, image2, alpha=0.5):
    image1_r, image2_r = align_images(image1, image2)
    return cv2.addWeighted(image1_r, alpha, image2_r, 1 - alpha, 0)


def overlay_images_max(image1, image2):
    image1_r, image2_r = align_images(image1, image2)
    return np.maximum(image1_r, image2_r)


def combine_feature_images(images):
    result = np.zeros_like(images[0])
    for image in images:
        mask_gray = np.logical_and(
            np.any(image > 0, axis=-1), np.any(result == 0, axis=-1)
        )
        mask_f = np.any(image > 10, axis=-1)
        mask_overlay = np.logical_or(mask_f, mask_gray)
        result[mask_overlay] = image[mask_overlay]
    return result


def remove_features(image, features, threshold=10):
    image_new = np.array(image)
    mask = np.zeros(image.shape[:2], dtype=bool)
    for feature in features:
        mask = np.maximum(mask, np.any(feature > threshold, axis=-1))

    mask_wing = np.any(image > threshold, axis=-1)
    dominant_color = get_dominant_colors(
        image[mask_wing],
        k=5,
        tolerance=50,
        black_threshold=threshold,
    )[0]
    inpaint_mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3))).astype(
        bool
    )
    image_new[mask] = dominant_color
    image_new[inpaint_mask] = dominant_color
    # image_new = cv2.inpaint(image_new, inpaint_mask, 1, cv2.INPAINT_TELEA)
    return image_new


def create_hybrid_image(
    image1, image2, max_features=3, max_extract_features=3, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    image1_r, image2_r = align_images(image1, image2)
    features1 = extract_features(image1_r, tolerance=70)[:max_extract_features]
    features2 = extract_features(image2_r, tolerance=70)[:max_extract_features]

    # select base wing canvas
    wing_base_idx = rng.choice([0, 1])

    if wing_base_idx == 0:
        base = image1_r
        base_features = features1
        wing_features = features2
    else:
        base = image2_r
        base_features = features2
        wing_features = features1

    # remove features from base
    hybrid = remove_features(base, base_features)

    # now add random features
    features = base_features + wing_features
    n_features = rng.integers(0, max_features + 1)
    chosen_features = rng.choice(
        np.arange(len(features)), n_features, replace=False
    )
    is_hybrid = False
    # if not all(
    #     [idx in chosen_features for idx in range(len(base_features))]
    # ):
    #     is_hybrid = True
    if any(chosen_features >= len(base_features)):
        is_hybrid = True

    hybrid_features = np.zeros_like(hybrid)
    for feature_idx in chosen_features:
        feature = features[feature_idx]
        mask = np.any(feature > 10, axis=-1)

        method = rng.integers(0, 2)  # 0: replace, 1: max, 2: weighted
        if method == 0:
            hybrid_features[mask] = feature[mask]
        elif method == 1:
            hybrid_features = np.maximum(hybrid_features, feature)
        elif method == 2:
            alpha = rng.uniform(0.2, 0.8)
            hybrid_features = cv2.addWeighted(
                hybrid_features, alpha, feature, 1 - alpha, 0
            )

    mask = np.any(hybrid_features > 10, axis=-1)
    hybrid[mask] = hybrid_features[mask]

    return hybrid, is_hybrid


def find_blobs(img_blob, dilation=10):
    binary = np.any(img_blob > 10, axis=-1).astype(np.uint8) * 255
    kernel = np.ones((dilation, dilation), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Find blobs
    return cv2.connectedComponentsWithStats(binary)


def seperate_blobs(img_blob):
    mask_blob = np.any(img_blob > 10, axis=-1)
    mask_gray = np.logical_and(~mask_blob, np.any(img_blob > 0, axis=-1))

    # Find blobs
    num_labels, labels, stats, centroids = find_blobs(img_blob, dilation=10)

    images = []
    for label in range(1, num_labels):
        mask = np.logical_and(labels == label, mask_blob)
        mask_new_gray = np.logical_and(labels != label, labels > 0)
        new_image = np.zeros_like(img_blob)
        new_image[mask] = img_blob[mask]
        new_image[mask_gray] = 10
        new_image[mask_new_gray] = 10
        images.append(new_image)

    return images


def compute_subspecies_probability(predictions, truth_matrix):
    """Compute the probability of belonging to each of the subspecies

    Parameters
    ----------
    predictions : np.ndarray
        Array of shape (n_samples, n_features) containing the predictions.
    truth_matrix : np.ndarray
        Array of shape (n_subspecies, n_features) containing the truth matrix.

    Returns
    -------
    subspecies_probability : np.ndarray
        The probability of belonging to each of the subspecies.
        Shape: (n_samples, n_subspecies)
    """
    n_samples, n_features = predictions.shape

    # shape: (n_samples, 1, n_features)
    predictions = predictions[:, None, :]

    # shape: (1, n_subspecies, n_features)
    truth_matrix = truth_matrix[None, ...]

    # shape: (n_samples, n_subspecies, n_features)
    res = predictions * truth_matrix + (1 - predictions) * (1 - truth_matrix)

    subspecies_probability = np.exp(np.sum(np.log(res), axis=-1))
    return subspecies_probability


def compute_anomaly(predictions, truth_matrix):
    """Compute the anomaly score for each sample.

    Parameters
    ----------
    predictions : np.ndarray
        Array of shape (n_samples, n_features) containing the predictions.
    truth_matrix : np.ndarray
        Array of shape (n_subspecies, n_features) containing the truth matrix.

    Returns
    -------
    anomaly_score : np.ndarray
        The anomaly score for each sample.
    """
    subspecies_probability = compute_subspecies_probability(
        predictions, truth_matrix
    )
    return 1 - np.max(subspecies_probability, axis=-1)


def compute_hybrid_probability(predictions_wings):
    """Compute the probability of a hybrid butterfly.

    We assume that this is the case if either all classes are near 0
    predicted scores, or if there are two classes with high predicted
    scores.
    No hybrid is assumed to only have 1 class with a high predicted score.

    Parameters
    ----------
    predictions_wings : np.ndarray
        Array of shape (n_samples, n_subspecies) containing the predictions.

    Returns
    -------
    subspecies_probability : np.ndarray
        The probability of belonging to each of the subspecies.
        Shape: (n_samples, n_subspecies)
    """
    n_samples, n_subspecies = predictions_wings.shape

    # sort the predictions and descending order
    # Shape: (n_samples, n_subspecies)
    sorted_predictions = np.sort(predictions_wings, axis=1)[:, ::-1]

    # define truth matrices
    # Shape: (n_subspecies)
    truth_matrix_single = np.r_[1, np.zeros(n_subspecies - 1)]
    truth_matrix_double = np.r_[1, 1, np.zeros(n_subspecies - 2)]
    truth_matrix_zero = np.zeros(n_subspecies)

    # Shape: (n_samples, n_subspecies)
    p_single = sorted_predictions * truth_matrix_single + (
        1 - sorted_predictions
    ) * (1 - truth_matrix_single)
    p_double = sorted_predictions * truth_matrix_double + (
        1 - sorted_predictions
    ) * (1 - truth_matrix_double)
    p_zero = sorted_predictions * truth_matrix_zero + (
        1 - sorted_predictions
    ) * (1 - truth_matrix_zero)

    # Shape: (n_samples)
    p_single = np.exp(np.sum(np.log(p_single), axis=-1))
    p_double = np.exp(np.sum(np.log(p_double), axis=-1))
    p_zero = np.exp(np.sum(np.log(p_zero), axis=-1))

    # Shape: (n_samples, 3)
    p_hybrid = np.c_[1 - p_single, p_zero]

    return np.max(p_hybrid, axis=-1)
