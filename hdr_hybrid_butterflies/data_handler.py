import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import albumentations as A
import multiprocessing as mp

from hdr_hybrid_butterflies import data_utils

try:
    import imagesize
except ImportError:
    print("imagesize not installed. Continuing without data handler support.")


class ImageProcessor:
    def __init__(
        self,
        detector_id="IDEA-Research/grounding-dino-tiny",
        segmenter_id="facebook/sam-vit-base",
        labels=["wings."],
        threshold=0.2,
        output_dim=(256, 256),
        segment_classifier=None,
        cnn_segmenter=None,
        cnn_segmenter_processor=None,
        p_erase=0.5,
        padding_size=100,
    ):
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.labels = labels
        self.threshold = threshold
        self.output_dim = output_dim
        self.segment_classifier = segment_classifier
        self.cnn_segmenter = cnn_segmenter
        self.cnn_segmenter_processor = cnn_segmenter_processor
        self.p_erase = p_erase
        self.padding_size = padding_size
        self.bbox_params = A.BboxParams(format="pascal_voc", label_fields=[])

        # Define augmentations
        p0 = 0.75
        p1 = 0.25
        p2 = 0.1
        self.augmentations = A.Compose(
            [
                A.AdditiveNoise(noise_type="uniform", p=p1),
                A.Blur(blur_limit=3, p=p2),
                A.HueSaturationValue(
                    hue_shift_limit=3,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=p1,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=p1
                ),
                A.CLAHE(clip_limit=2.0, p=p2),
                A.RGBShift(
                    r_shift_limit=3, g_shift_limit=3, b_shift_limit=3, p=p1
                ),
                A.RandomToneCurve(p=p1, scale=0.05),
                A.RandomGamma(p=p1),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=p2),
                A.HorizontalFlip(p=p2),
                A.VerticalFlip(p=p2),
                A.Transpose(p=p2),
                A.OpticalDistortion(p=p0),
                A.GridDistortion(distort_limit=0.1, p=p0),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.0,
                    rotate_limit=15,
                    p=p0,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                ),
                A.Erasing(p=self.p_erase, fill=0, scale=(0.02, 0.33)),
            ],
            bbox_params=self.bbox_params,
        )

        self.padding = A.Compose(
            [A.Pad(self.padding_size, p=1.0)], bbox_params=self.bbox_params
        )
        self.to_gray = A.Compose(
            [A.ToGray(p=1.0)], bbox_params=self.bbox_params
        )
        self.resize = A.Compose(
            [A.Resize(*output_dim)], bbox_params=self.bbox_params
        )

    def __call__(self, image, mask_only=False, grayscale=False, via_cnn=False):
        """Process in image into a format suitable for the model

        Parameters
        ----------
        image : PIL.Image
            The image to process
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        via_cnn : bool
            If True, use the CNN segmenter instead
            of the grounded segmenter.

        Returns
        -------
        np.ndarray
            An array of lower wing segments
        np.ndarray
            An array of upper wing segments
        """
        image = ImageOps.exif_transpose(image)

        if via_cnn:
            lower_segments, upper_segments = self.extract_segments_via_cnn(
                image
            )
        else:
            # extract segments from image
            _, _, segments, _ = self._raw_segments(image)

            # classify segments
            _, upper_list, lower_list = self.classify_segments(segments)

            upper_segments = [segments[idx] for idx in upper_list]
            lower_segments = [segments[idx] for idx in lower_list]

        # process the segments
        def process_segments(segments):
            return [
                self.augment_image(
                    segment,
                    mask_only=mask_only,
                    grayscale=grayscale,
                    apply_augmentations=False,
                )[0]
                for segment in segments
            ]

        upper_segments = process_segments(upper_segments)
        lower_segments = process_segments(lower_segments)

        # add zeros to get at least 2 segments
        out_shape = (self.output_dim[0], self.output_dim[1], 3)
        zeros = np.zeros(out_shape, dtype=np.uint8)

        while len(upper_segments) < 2:
            upper_segments.append(zeros)

        while len(lower_segments) < 2:
            lower_segments.append(zeros)

        return np.stack(lower_segments), np.stack(upper_segments)

    def extract_segments_via_cnn(self, image, reduction_factor=4):
        """Extract segments via a CNN

        Parameters
        ----------
        image : PIL.Image
            The image to extract segments from

        Returns
        -------
        List[np.ndarray]
            An list of lower wing segments
        List[np.ndarray]
            An list of upper wing segments
        """
        import timeit

        t_0 = timeit.default_timer()
        orig_image = np.asarray(image)
        img = np.array(image)

        if reduction_factor > 1:
            new_size = np.array(img.shape[:2]) // reduction_factor
            img = A.Resize(*new_size)(image=img)["image"]

        assert (
            self.cnn_segmenter_processor.padding_size == 0
        ), "Padding not supported"

        # augment image
        # Image is now in square format
        img_aug, _ = self.cnn_segmenter_processor.augment_image(
            img,
            mask=None,
            mask_only=False,
            grayscale=False,
            apply_augmentations=False,
        )

        # apply segmentation
        t_1 = timeit.default_timer()
        probs = self.cnn_segmenter.probabilities(img_aug[None, ...])
        t_2 = timeit.default_timer()
        contours, masks, scores, boxes = self.extract_polygons(probs[0])
        t_3 = timeit.default_timer()

        # scale back up to squared img.shape
        square_dim = max(img.shape[:2])
        scale_x = square_dim / img_aug.shape[0]
        scale_y = square_dim / img_aug.shape[1]
        for contour in contours:
            contour[:, :, 0] = contour[:, :, 0] * scale_x
            contour[:, :, 1] = contour[:, :, 1] * scale_y

        # undo padding
        contours = [
            self.contour_undo_square_padding(img.shape[:2], contour)
            for contour in contours
        ]

        # undo reduction_factor
        if reduction_factor > 1:
            for contour in contours:
                contour[:, :, 0] = contour[:, :, 0] * reduction_factor
                contour[:, :, 1] = contour[:, :, 1] * reduction_factor

        # extract segments
        upper_segments = []
        lower_segments = []

        t_4 = timeit.default_timer()
        for contour, score in zip(contours, scores):
            # Extract the vertices of the contour
            polygon = contour.reshape(-1, 2).tolist()

            # Create an empty mask
            mask_i = np.zeros(orig_image.shape[:2], dtype=np.uint8)

            # Convert polygon to an array of points
            pts = np.array(polygon, dtype=np.int32)

            # Fill the polygon with white color (255)
            cv2.fillPoly(mask_i, [pts], color=(255,))

            # extract bounding box segment
            x_pos, y_pos, width, height = cv2.boundingRect(contour)
            x_min = y_pos
            x_max = y_pos + height
            y_min = x_pos
            y_max = x_pos + width

            # select segment
            segment = np.array(orig_image[x_min:x_max, y_min:y_max])
            mask_i = mask_i[x_min:x_max, y_min:y_max]

            # apply mask to image
            segment[mask_i == 0] = 0

            # classify the segment
            if np.argmax(score) == 1:
                lower_segments.append(segment)
            elif np.argmax(score) == 2:
                upper_segments.append(segment)

        t_5 = timeit.default_timer()
        print(
            f"  Times: {t_1-t_0:.2f}, {t_2-t_1:.2f}, "
            f"{t_3-t_2:.2f}, {t_4-t_3:.2f}, {t_5-t_4:.2f}"
        )
        return lower_segments, upper_segments

    def extract_polygons(self, prob_array, threshold=0.5):
        """Extract polygons from a probability array

        Parameters
        ----------
        prob_array : np.ndarray
            The predicted probabilities for each pixel.
        threshold : float
            The threshold to use for extracting polygons.

        Returns
        -------
        List[np.ndarray]
            A list of contours.
        List[np.ndarray]
            A list of binary masks for the extracted polygons.
        List[np.ndarray]
            A list of scores for the extracted polygons.
        List[np.ndarray]
            A list of bounding boxes for the extracted polygons.
            Each bounding box is given by (x_min, y_min, x_max, y_max).
        """
        mask_wing = (
            (np.sum(prob_array[:, :, 1:3], axis=-1) > threshold) * 255
        ).astype(np.uint8)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            mask_wing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        masks = []
        scores = []
        boxes = []
        chosen_contours = []
        for contour in contours[:4]:
            # throw out too small contours
            if cv2.contourArea(contour) < 0.003 * mask_wing.size:
                continue

            # Extract the vertices of the contour
            polygon = contour.reshape(-1, 2).tolist()

            # Create an empty mask
            mask_i = np.zeros(prob_array.shape, dtype=np.uint8)

            # Convert polygon to an array of points
            pts = np.array(polygon, dtype=np.int32)

            # Fill the polygon with white color (255)
            cv2.fillPoly(mask_i, [pts], color=(255,))

            # classify the polygon
            mask_pixels = np.any(mask_i > 0, axis=-1)
            probs_polygon = np.mean(prob_array[mask_pixels], axis=0)

            # throw out likely background polygons
            if np.argmax(probs_polygon) == 0:
                continue

            # extract bounding box segment
            x_pos, y_pos, width, height = cv2.boundingRect(contour)
            x_min = y_pos
            x_max = y_pos + height
            y_min = x_pos
            y_max = x_pos + width

            chosen_contours.append(contour)
            masks.append(mask_i)
            scores.append(probs_polygon)
            boxes.append((x_min, y_min, x_max, y_max))

        return chosen_contours, masks, scores, boxes

    def classify_segments(self, segments):
        """Classify segments

        Parameters
        ----------
        segments : List[np.ndarray]
            A list of segments

        Returns
        -------
        List[int]
            A list of classifications
        """
        # transform segments for model input
        processed_segments = np.stack(
            [
                self.augment_image(
                    segment,
                    mask_only=True,
                    grayscale=False,
                    apply_augmentations=False,
                )[0]
                for segment in segments
            ],
            axis=0,
        )

        # apply classifier to segments
        if self.segment_classifier is None:
            raise ValueError("Segment classifier must be specified!")

        predictions = np.array(self.segment_classifier(processed_segments))

        # 0: noise, 1: lower, 2: upper
        classified_type = np.argmax(predictions, axis=1)
        upper_list = []
        lower_list = []
        for idx, pred_class in enumerate(classified_type):
            if pred_class == 2:
                upper_list.append(idx)
            elif pred_class == 1:
                lower_list.append(idx)

        # remove segments with low confidence if more than 2 segments
        if len(upper_list) > 2:
            sorted_idx = np.argsort(predictions[upper_list, 2])
            upper_list = [upper_list[idx] for idx in sorted_idx[-2:]]

        if len(lower_list) > 2:
            sorted_idx = np.argsort(predictions[lower_list, 1])
            lower_list = [lower_list[idx] for idx in sorted_idx[-2:]]

        # make sure at least one segment is selected
        # Don't care if it already used twice
        if len(upper_list) == 0:
            upper_list = [np.argmax(predictions[:, 2])]

        if len(lower_list) == 0:
            lower_list = [np.argmax(predictions[:, 1])]

        return predictions, upper_list, lower_list

    def grounded_segmentation(self, image):
        """Perform grounded segmentation on an image

        Parameters
        ----------
        image : PIL.Image
            The image to perform grounded segmentation on

        Returns
        -------
        numpy.ndarray
            The image array.
        List[DetectionResult]
            A list of detections.
        """
        from hdr_hybrid_butterflies.dino_sam import dino_sam

        return dino_sam.grounded_segmentation(
            image=image,
            labels=self.labels,
            threshold=self.threshold,
            polygon_refinement=True,
            detector_id=self.detector_id,
            segmenter_id=self.segmenter_id,
        )

    def _raw_segments(self, image):
        """Get raw segments from an image

        Parameters
        ----------
        image : PIL.Image
            The image to get raw segments from

        Returns
        -------
        np.ndarray
            The image array.
        List[DetectionResult]
            A list of detections.
        List[np.ndarray]
            A list of cropped and masked image segments.
        List[float]
            A list of scores for each segment.
        """
        image_array, detections = self.grounded_segmentation(image)

        segments = []
        scores = []
        for detection in detections:
            segment = np.array(image_array)
            segment[detection.mask == 0] = 0

            segment = segment[
                detection.box.ymin : detection.box.ymax,
                detection.box.xmin : detection.box.xmax,
            ]
            segments.append(segment)
            scores.append(detection.score)
        return image_array, detections, segments, scores

    def _raw_segmentation_labels(self, image):
        """Get raw segmentation labels from an image

        Parameters
        ----------
        image : PIL.Image
            The image to get raw segments from

        Returns
        -------
        np.ndarray
            The image array.
        np.ndarray
            The mask array. RGB channels:
                Red / 0: noise
                Green / 1: lower wing
                Blue / 2: upper wing
        List[BoundingBox]
            A list of bounding boxes for each upper wing.
        List[BoundingBox]
            A list of bounding boxes for each lower wing.
        """
        # ensure image is in landscape orientation
        if image.size[0] < image.size[1]:
            image = image.transpose(Image.ROTATE_90)

        assert image.size[0] >= image.size[1], image.size

        # extract segments from image
        image_array, detections, segments, _ = self._raw_segments(image)

        # classify segments
        _, upper_list, lower_list = self.classify_segments(segments)

        mask_array = np.zeros(image_array.shape, dtype=np.uint8)
        mask_upper = np.any([detections[i].mask for i in upper_list], axis=0)
        mask_lower = np.any([detections[i].mask for i in lower_list], axis=0)
        mask_array[np.logical_not(mask_lower | mask_upper), 0] = 255
        mask_array[mask_lower, 1] = 255
        mask_array[mask_upper, 2] = 255

        boxes_upper = [detections[i].box for i in upper_list]
        boxes_lower = [detections[i].box for i in lower_list]

        return image_array, mask_array, boxes_upper, boxes_lower

    def pad_to_square(self, image):
        """Pad image and mask to square dimension

        Parameters
        ----------
        image : np.ndarray
            The image to pad.

        Returns
        -------
        np.ndarray
            The padded image.
        """
        # pad image to square
        square_size = max(image.shape[:2])
        pad_axis = 0 if image.shape[0] < image.shape[1] else 1
        pad_size = square_size - image.shape[pad_axis]
        pad_half = pad_size // 2
        new_shape = [square_size, square_size] + list(image.shape[2:])

        new_image = np.zeros(new_shape, dtype=np.uint8)

        if pad_axis == 0:
            new_image[pad_half : pad_half + image.shape[0]] = image
        else:
            new_image[:, pad_half : pad_half + image.shape[1]] = image

        return new_image

    def undo_square_padding(self, original_shape, image):
        """Undo square padding

        Parameters
        ----------
        original_shape : Tuple[int, int]
            The original shape of the image.
        image : np.ndarray
            The padded image.
        mask : np.ndarray
            The padded mask.

        Returns
        -------
        np.ndarray
            The unpadded image.
        np.ndarray
            The unpadded mask.
        """
        square_size = max(original_shape)
        pad_axis = 0 if original_shape[0] < original_shape[1] else 1
        pad_size = square_size - original_shape[pad_axis]
        pad_half = pad_size // 2

        if pad_axis == 0:
            new_image = image[pad_half : pad_half + original_shape[0]]
        else:
            new_image = image[:, pad_half : pad_half + original_shape[1]]

        return new_image

    def contour_undo_square_padding(self, original_shape, contour):
        """Undo square padding for a defined contour

        Parameters
        ----------
        original_shape : Tuple[int, int]
            The original shape of the image.
        contour : np.ndarray
            The contour in the padded image.

        Returns
        -------
        np.ndarray
            The contour in the unpadded image.
        """
        square_size = max(original_shape)
        pad_axis = 0 if original_shape[0] < original_shape[1] else 1
        pad_size = square_size - original_shape[pad_axis]
        pad_half = pad_size // 2

        if pad_axis == 0:
            new_contour = np.array(contour) - np.array([0, pad_half])
        else:
            new_contour = np.array(contour) - np.array([pad_half, 0])

        return new_contour

    def bbox_undo_square_padding(self, original_shape, box):
        """Undo square padding for bounding boxes

        Parameters
        ----------
        original_shape : Tuple[int, int]
            The original shape of the image.
        box : List[int]
            The bounding box in the padded image.
            [x_min, y_min, x_max, y_max]

        Returns
        -------
        List[int]
            The bounding box in the unpadded image.
        """
        square_size = max(original_shape)
        pad_axis = 0 if original_shape[0] < original_shape[1] else 1
        pad_size = square_size - original_shape[pad_axis]
        pad_half = pad_size // 2

        if pad_axis == 0:
            new_box = [box[0] - pad_half, box[1], box[2] - pad_half, box[3]]
        else:
            new_box = [box[0], box[1] - pad_half, box[2], box[3] - pad_half]

        return new_box

    def augment_image(
        self,
        image,
        mask=None,
        mask_only=False,
        grayscale=False,
        apply_augmentations=True,
    ):
        """Augment an image

        Parameters
        ----------
        image : cv2.Image | PIL.Image | np.ndarray
            The image to augment.
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        apply_augmentations : bool
            If True, apply augmentations.
            If False, only resize the image.

        Returns
        -------
        np.ndarray
            The augmented image.
        np.ndarray
            The augmented mask.
        """
        image = np.asarray(image)
        if mask is None:
            reapply_mask = True
            mask = np.asarray(np.any(image != 0, axis=-1), dtype=np.uint8)
        else:
            reapply_mask = False

        # apply padding
        if self.padding_size > 0:
            padded = self.padding(image=image, mask=mask)
            image = padded["image"]
            mask = padded["mask"]

        if apply_augmentations:
            augmentation = self.augmentations(
                image=image,
                mask=mask,
            )
        else:
            # identity augmentation
            augmentation = {"image": image, "mask": mask}

        image = augmentation["image"]
        mask = augmentation["mask"]

        # re-apply mask
        if reapply_mask:
            image = image * mask[..., None]

        # pad image to square
        new_image = self.pad_to_square(image)
        new_mask = self.pad_to_square(mask)

        # resize to output dim
        resized = self.resize(image=new_image, mask=new_mask)
        new_image = resized["image"]
        new_mask = resized["mask"]

        if mask_only:
            mask_new_image = np.any(new_image != 0, axis=-1)
            new_image[mask_new_image] = 255

        if grayscale:
            new_image = self.to_gray(image=new_image)["image"]
        return new_image, new_mask


class SegmentDataHandler:
    def __init__(
        self,
        meta_data_path,
        data_dir_upper,
        data_dir_lower,
        data_dir_noise,
        image_processor,
        skip_hybrid=True,
        test_split=0.05,
        seed=42,
    ):
        """Initialize the data handler

        Parameters
        ----------
        meta_data_path : str
            Path to the meta data file of the original dataset.
        data_dir_upper : str
            Path to the directory containing the images
            of the upper wing.
        data_dir_lower : str
            Path to the directory containing the images
            of the lower wing.
        data_dir_noise : str
            Path to the directory containing the images
            of noise.
        image_processor : ImageProcessor
            The image processor to use.
        skip_hybrid : bool
            If True, skip hybrid images.
        test_split : float
            Fraction of the data to use for testing.
        seed : int
            Seed for random number generator
        """
        self.processes = []
        self.rng = np.random.default_rng(seed)
        self.data_dir = {
            "upper": os.path.abspath(data_dir_upper),
            "lower": os.path.abspath(data_dir_lower),
            "noise": os.path.abspath(data_dir_noise),
        }
        self.test_split = test_split
        self.image_processor = image_processor
        self.skip_hybrid = skip_hybrid

        self.df_meta_original = pd.read_csv(meta_data_path)

        # create meta dataframe
        self.df_meta = {
            "filename": [],
            "label": [],
            "score": [],
            "width": [],
            "height": [],
            "ratio": [],
            "subspecies": [],
            "parent_subspecies_1": [],
            "parent_subspecies_2": [],
            "CAMID": [],
        }
        for data_dir, label in zip(
            [data_dir_upper, data_dir_lower, data_dir_noise],
            ["upper", "lower", "noise"],
        ):
            file_list = sorted(glob(os.path.join(data_dir, "*.jpg")))
            for filename in file_list:
                base_name = os.path.basename(filename)
                cam_id_orig = "".join(base_name.split("_")[:-2])
                mask = self.df_meta_original["CAMID"] == cam_id_orig
                row_original = self.df_meta_original[mask]
                if len(row_original) == 0:
                    print(f"Warning: {cam_id_orig} not found in meta data.")
                    continue
                assert len(row_original) == 1, (cam_id_orig, row_original)

                # skip hybrid images
                if self.skip_hybrid:
                    if not np.isfinite(row_original["subspecies"].iloc[0]):
                        continue

                # add meta info
                for key in [
                    "subspecies",
                    "parent_subspecies_1",
                    "parent_subspecies_2",
                    "CAMID",
                ]:
                    self.df_meta[key].append(row_original[key].iloc[0])

                score = float(filename.split("_")[-1][1:-4])
                width, height = imagesize.get(filename)
                ratio = width / height
                score = float(filename.split("_")[-1][1:-4])

                self.df_meta["filename"].append(base_name)
                self.df_meta["label"].append(label)
                self.df_meta["score"].append(score)
                self.df_meta["width"].append(width)
                self.df_meta["height"].append(height)
                self.df_meta["ratio"].append(ratio)

        self.df_meta = pd.DataFrame(self.df_meta)

        # randomize order
        self.df_meta = self.df_meta.sample(frac=1, random_state=self.rng)
        self.n_samples = len(self.df_meta)
        self.n_samples_train = int(self.n_samples * (1 - self.test_split))
        self.n_samples_test = self.n_samples - self.n_samples_train

        self.indices = np.arange(self.n_samples)

    def load_data(self, index):
        """Load image and meta data

        Parameters
        ----------
        index : int
            Index of the sample to load

        Returns
        -------
        img : PIL.Image
            The loaded image
        row : pd.Series
            The meta data of the loaded image
        """
        row = pd.Series(self.df_meta.iloc[index])
        img_path = os.path.join(
            self.data_dir[row["label"]],
            row["filename"],
        )
        img = Image.open(img_path)
        return img, row

    def load_by_name(self, name):
        """Load image and meta data by name

        Parameters
        ----------
        name : str
            Name of the sample to load

        Returns
        -------
        img : PIL.Image
            The loaded image
        row : pd.Series
            The meta data of the loaded image
        """
        row = self.df_meta[self.df_meta["filename"] == name].iloc[0]
        img_path = os.path.join(self.data_dir[row["label"]], row["filename"])
        img = Image.open(img_path)
        return img, row

    def load_df_meta_segments_for_camid(self, camid, labels=None):
        """Load segments meta info for an original image

        Parameters
        ----------
        camid : str
            The camera id of the original image
        labels : List[str]
            List of labels to include.
            Options are ["upper", "lower", "noise"].
            If None, include all.

        Returns
        -------
        df_meta_segments : pd.DataFrame
            Meta data of the segments belonging to the
            original image with the specified camid.
        """
        mask = [camid in name for name in self.df_meta["filename"]]
        if labels is not None:
            mask &= self.df_meta["label"].isin(labels)
        df_meta_segments = self.df_meta[mask]

        # remove duplicates based on label, score, width, height
        df_meta_segments = df_meta_segments.drop_duplicates(
            subset=["label", "score", "width", "height"]
        )

        return df_meta_segments

    def __call__(
        self,
        mask_only=False,
        grayscale=False,
        seed=None,
        training=True,
        sample_weights=None,
        apply_augmentations=True,
    ):
        """Load a random image and augment it

        Parameters
        ----------
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        seed : int
            Seed for random number generator
        training : bool
            If True, sample from the training set.
            Otherwise, sample from the test set.
        sample_weights : np.ndarray
            Weights for sampling.
            If None, use uniform sampling.
        apply_augmentations : bool
            If True, apply augmentations.

        Returns
        -------
        img_aug : np.ndarray
            The augmented image
        row : pd.Series
            The meta data of the loaded image
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        # sample random image
        if training:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train)
            else:
                index = rng.choice(
                    self.indices[: self.n_samples_train],
                    p=sample_weights[: self.n_samples_train],
                )
        else:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train, self.n_samples)
            else:
                index = rng.choice(
                    self.indices[self.n_samples_train :],
                    p=sample_weights[self.n_samples_train :],
                )
        img, row = self.load_data(index)

        # augment image
        img_aug, _ = self.image_processor.augment_image(
            img,
            mask_only=mask_only,
            grayscale=grayscale,
            apply_augmentations=apply_augmentations,
        )
        return img_aug, row

    def segmentation_labels(self, row):
        """Generate training labels for segmentation

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image
        """
        if row["label"] == "noise":
            label = 0
        elif row["label"] == "lower":
            label = 1
        elif row["label"] == "upper":
            label = 2
        else:
            raise ValueError(f"Unknown label: {row['label']}")
        return label

    def hybrid_labels(self, row):
        """Generate training labels for hybrid images

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image
        """
        if np.isfinite(row["subspecies"]):
            label = 0
        else:
            label = 1
        return label

    # setting this dynamically didn't seem to work, so copy paste it is.
    # choosing not to go with an argument here to maint compatibility with the
    # the existing code
    def label_subspecies_00(self, row):
        return row["subspecies"] == 0

    def label_subspecies_01(self, row):
        return row["subspecies"] == 1

    def label_subspecies_02(self, row):
        return row["subspecies"] == 2

    def label_subspecies_03(self, row):
        return row["subspecies"] == 3

    def label_subspecies_04(self, row):
        return row["subspecies"] == 4

    def label_subspecies_05(self, row):
        return row["subspecies"] == 5

    def label_subspecies_06(self, row):
        return row["subspecies"] == 6

    def label_subspecies_07(self, row):
        return row["subspecies"] == 7

    def label_subspecies_08(self, row):
        return row["subspecies"] == 8

    def label_subspecies_09(self, row):
        return row["subspecies"] == 9

    def label_subspecies_10(self, row):
        return row["subspecies"] == 10

    def label_subspecies_11(self, row):
        return row["subspecies"] == 11

    def label_subspecies_12(self, row):
        return row["subspecies"] == 12

    def label_subspecies_13(self, row):
        return row["subspecies"] == 13

    def label_subspecies_14(self, row):
        return row["subspecies"] == 14

    def get_generator(
        self,
        batch_size=32,
        queue_size=32,
        labels_func_name="segmentation_labels",
        mask_only=False,
        grayscale=False,
        training=True,
        balanced_loading=False,
        n_jobs=1,
    ):
        """Get a data generator

        Parameters
        ----------
        batch_size : int
            Batch size.
        queue_size : int
            Size of the queue.
        labels_func_name : str
            The name of the function to generate labels.
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        training : bool
            If True, sample from the training set.
            Otherwise, sample from the test set.
        balanced_loading : bool
            If True, load samples balanced.
        n_jobs : int
            Number of processes to use

        Returns
        -------
        generator : function
            A function that generates augmented images
        """
        queue = mp.Manager().Queue(maxsize=queue_size)

        label_creater = getattr(self, labels_func_name)

        # get sample weights
        if balanced_loading:
            if training:
                slice_data = slice(0, self.n_samples_train)
            else:
                slice_data = slice(self.n_samples_train, self.n_samples)
            labels = []
            for row in self.df_meta.iloc[slice_data].iterrows():
                labels.append(label_creater(row[1]))
            labels = np.array(labels)
            sample_weights = np.zeros(len(labels))
            print(
                f"Label occurrences: {np.unique(labels, return_counts=True)}"
            )
            for class_label in np.unique(labels):
                mask = labels == class_label
                class_weight = len(labels) / np.sum(mask)
                sample_weights[mask] = len(labels) / np.sum(mask)
                print(f"Class {class_label} weight: {class_weight}")

            sample_weights = sample_weights / sample_weights.sum()
        else:
            sample_weights = None

        def worker(seed):
            rng = np.random.default_rng(seed)
            while True:
                img_aug, row = self(
                    mask_only=mask_only,
                    grayscale=grayscale,
                    training=training,
                    sample_weights=sample_weights,
                    seed=rng.integers(2**32),
                )
                queue.put((img_aug, row))

        for i in range(n_jobs):
            process = mp.Process(target=worker, args=(i,))
            process.start()
            self.processes.append(process)

        def generator():
            while True:
                batch_images = []
                batch_labels = []
                while len(batch_images) < batch_size:
                    img_aug, row = queue.get()
                    batch_images.append(img_aug)
                    batch_labels.append(label_creater(row))
                images = np.stack(batch_images, axis=0)
                labels = np.array(batch_labels)
                yield images, labels

        return generator()

    def __del__(self):
        for process in self.processes:
            process.terminate()


class WingSegmentDataHandler(SegmentDataHandler):
    def __init__(
        self,
        meta_data_path,
        data_dir_upper,
        data_dir_lower,
        image_processor,
        skip_hybrid=True,
        test_split=0.05,
        seed=42,
    ):
        super().__init__(
            meta_data_path=meta_data_path,
            data_dir_upper=data_dir_upper,
            data_dir_lower=data_dir_lower,
            data_dir_noise="dummy_non_existing",
            image_processor=image_processor,
            skip_hybrid=skip_hybrid,
            test_split=test_split,
            seed=seed,
        )

    def label_subspecies(self, row):
        """Generate training labels for subspecies

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image
        """
        return row["subspecies"]

    def __call__(
        self,
        mask_only=False,
        grayscale=False,
        seed=None,
        training=True,
        sample_weights=None,
        sample_random_segments=True,
        apply_augmentations=True,
        p_drop_segment=0.1,
    ):
        """Load a wing consisting of 4 (random) segments and augment it

        Parameters
        ----------
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        seed : int
            Seed for random number generator
        training : bool
            If True, sample from the training set.
            Otherwise, sample from the test set.
        sample_weights : np.ndarray
            Weights for sampling.
            If None, use uniform sampling.
        sample_random_segments : bool
            If True, sample random segments to combine into
            a wing for the given subspecies.
            If False, sample segments from the same camera id.
        apply_augmentations : bool
            If True, apply augmentations.
        p_drop_segment : float
            Probability to drop a segment for each of
            the upper and lower wing.
            No dropping if 0.

        Returns
        -------
        img_aug : np.ndarray
            The augmented image
        row : pd.Series
            The meta data of the loaded image
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        # sample random image
        if training:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train)
            else:
                index = rng.choice(
                    self.indices[: self.n_samples_train],
                    p=sample_weights[: self.n_samples_train],
                )
        else:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train, self.n_samples)
            else:
                index = rng.choice(
                    self.indices[self.n_samples_train :],
                    p=sample_weights[self.n_samples_train :],
                )

        # get meta data for the chosen segment
        row = pd.Series(self.df_meta.iloc[index])

        # subspecies of the image
        row_original = self.df_meta_original[
            self.df_meta_original["CAMID"] == row["CAMID"]
        ].iloc[0]

        if sample_random_segments:
            subspecies = row["subspecies"]
            mask = self.df_meta["subspecies"] == subspecies
            df_upper = self.df_meta[mask & (self.df_meta["label"] == "upper")]
            df_lower = self.df_meta[mask & (self.df_meta["label"] == "lower")]
            df_upper = df_upper.sample(n=2, random_state=rng)
            df_lower = df_lower.sample(n=2, random_state=rng)
        else:
            camid = row["CAMID"]
            df_camid = self.load_df_meta_segments_for_camid(camid)
            df_upper = df_camid[df_camid["label"] == "upper"]
            df_lower = df_camid[df_camid["label"] == "lower"]

            assert len(df_upper) <= 2, df_upper
            assert len(df_lower) <= 2, df_lower

        if p_drop_segment > 0:
            if rng.random() < p_drop_segment:
                df_upper = df_upper.iloc[:1]
            if rng.random() < p_drop_segment:
                df_lower = df_lower.iloc[:1]

        segments_upper = [
            self.load_by_name(name)[0] for name in df_upper["filename"]
        ]
        segments_lower = [
            self.load_by_name(name)[0] for name in df_lower["filename"]
        ]

        # augment segments
        segments_upper = [
            self.image_processor.augment_image(
                segment,
                mask_only=mask_only,
                grayscale=grayscale,
                apply_augmentations=apply_augmentations,
            )[0]
            for segment in segments_upper
        ]
        segments_lower = [
            self.image_processor.augment_image(
                segment,
                mask_only=mask_only,
                grayscale=grayscale,
                apply_augmentations=apply_augmentations,
            )[0]
            for segment in segments_lower
        ]

        if len(segments_upper) == 1:
            segments_upper.append(np.zeros_like(segments_upper[0]))
        if len(segments_lower) == 1:
            segments_lower.append(np.zeros_like(segments_lower[0]))

        wing = np.stack(
            segments_upper + segments_lower,
            axis=0,
        )

        return wing, row_original


class HybridStitcherWingSegmentDataHandler(SegmentDataHandler):
    def __init__(
        self,
        meta_data_path,
        data_dir_upper,
        data_dir_lower,
        image_processor,
        skip_hybrid=True,
        test_split=0.05,
        seed=42,
    ):
        super().__init__(
            meta_data_path=meta_data_path,
            data_dir_upper=data_dir_upper,
            data_dir_lower=data_dir_lower,
            data_dir_noise="dummy_non_existing",
            image_processor=image_processor,
            skip_hybrid=skip_hybrid,
            test_split=test_split,
            seed=seed,
        )

        # Defines combinations of Upper: lower subspecies
        # that will generate unique hybrid images
        self.hybrid_subspecies_combinations = {
            0: [1, 2, 3, 4, 10, 11],
            1: [0, 2, 5, 6, 7, 8, 9, 12, 13],
            2: [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            3: [0, 2, 5, 6, 7, 8, 9, 12, 13],
            4: [0, 2, 5, 6, 7, 8, 9, 12, 13],
            5: [1, 2, 3, 4, 10, 11],
            6: [0, 1, 2, 3, 4, 7, 9, 10, 11, 13],
            7: [2, 5, 6, 8, 12],
            8: [0, 1, 2, 3, 4, 7, 9, 10, 11, 13],
            9: [1, 2, 3, 4, 5, 6, 8, 10, 11, 12],
            10: [2, 5, 6, 8, 12],
            11: [0, 2, 5, 6, 7, 8, 9, 12, 13],
            12: [0, 1, 2, 3, 4, 7, 9, 10, 11, 13],
            13: [2, 5, 6, 8, 12],
        }

    def label_hybrid(self, row):
        """Generate training labels for hybrid images
        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True if hybrid, False if non-hybrid.
        """
        return row["is_hybrid"]

    def __call__(
        self,
        mask_only=False,
        grayscale=False,
        seed=None,
        training=True,
        sample_weights=None,
        sample_random_segments=True,
        apply_augmentations=True,
        p_drop_segment=0.1,
    ):
        """Load a wing consisting of 4 (random) segments and augment it

        Parameters
        ----------
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        seed : int
            Seed for random number generator
        training : bool
            If True, sample from the training set.
            Otherwise, sample from the test set.
        sample_weights : np.ndarray
            Weights for sampling.
            If None, use uniform sampling.
        sample_random_segments : bool
            If True, sample random segments to combine into
            a wing for the given subspecies.
            If False, sample segments from the same camera id.
        apply_augmentations : bool
            If True, apply augmentations.
        p_drop_segment : float
            Probability to drop a segment for each of
            the upper and lower wing.
            No dropping if 0.

        Returns
        -------
        img_aug : np.ndarray
            The augmented image
        row : pd.Series
            The meta data of the loaded image
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        # sample random image
        if training:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train)
            else:
                index = rng.choice(
                    self.indices[: self.n_samples_train],
                    p=sample_weights[: self.n_samples_train],
                )
        else:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train, self.n_samples)
            else:
                index = rng.choice(
                    self.indices[self.n_samples_train :],
                    p=sample_weights[self.n_samples_train :],
                )

        # get meta data for the chosen segment
        row = pd.Series(self.df_meta.iloc[index])

        # subspecies of the image
        row_original = self.df_meta_original[
            self.df_meta_original["CAMID"] == row["CAMID"]
        ].iloc[0]

        is_hybrid = rng.choice([True, False])
        if is_hybrid:
            subspecies = rng.choice(
                self.hybrid_subspecies_combinations[row["subspecies"]]
            )
        else:
            subspecies = row["subspecies"]

        mask_upper = self.df_meta["subspecies"] == row["subspecies"]
        mask_lower = self.df_meta["subspecies"] == subspecies
        df_upper = self.df_meta[
            mask_upper & (self.df_meta["label"] == "upper")
        ]
        df_lower = self.df_meta[
            mask_lower & (self.df_meta["label"] == "lower")
        ]
        df_upper = df_upper.sample(n=2, random_state=rng)
        df_lower = df_lower.sample(n=2, random_state=rng)

        row_original["is_hybrid"] = is_hybrid

        if p_drop_segment > 0:
            if rng.random() < p_drop_segment:
                df_upper = df_upper.iloc[:1]
            if rng.random() < p_drop_segment:
                df_lower = df_lower.iloc[:1]

        segments_upper = [
            self.load_by_name(name)[0] for name in df_upper["filename"]
        ]
        segments_lower = [
            self.load_by_name(name)[0] for name in df_lower["filename"]
        ]

        # augment segments
        segments_upper = [
            self.image_processor.augment_image(
                segment,
                mask_only=mask_only,
                grayscale=grayscale,
                apply_augmentations=apply_augmentations,
            )[0]
            for segment in segments_upper
        ]
        segments_lower = [
            self.image_processor.augment_image(
                segment,
                mask_only=mask_only,
                grayscale=grayscale,
                apply_augmentations=apply_augmentations,
            )[0]
            for segment in segments_lower
        ]

        if len(segments_upper) == 1:
            segments_upper.append(np.zeros_like(segments_upper[0]))
        if len(segments_lower) == 1:
            segments_lower.append(np.zeros_like(segments_lower[0]))

        wing = np.stack(
            segments_upper + segments_lower,
            axis=0,
        )

        return wing, row_original


class UpperWingDataHandler(SegmentDataHandler):
    def __init__(
        self,
        meta_data_path,
        data_dir_upper,
        image_processor,
        skip_hybrid=True,
        test_split=0.05,
        seed=42,
    ):
        super().__init__(
            meta_data_path=meta_data_path,
            data_dir_upper=data_dir_upper,
            data_dir_lower="dummy_non_existing",
            data_dir_noise="dummy_non_existing",
            image_processor=image_processor,
            skip_hybrid=skip_hybrid,
            test_split=test_split,
            seed=seed,
        )
        self.set_feature_definitions()

    def set_feature_definitions(self):
        self.feature_definitions = {
            0: [1, 2, 13],
            1: [0, 5, 6, 8, 12],
            2: [0, 5],
            3: [2],
            4: [3],
            5: [4],
            6: [7, 10, 13],
            7: [6, 9],
            8: [8],
            9: [9],
            10: [11],
            11: [12],
        }

    def labels_feature(self, row, feature_num):
        """Generate training labels

        Parameters
        ----------
        row : pd.Series
            Meta data of the image
        feature_num : int
            The feature number to compute the label for.

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature is present, False (0) otherwise.
        """
        return row["subspecies"] in self.feature_definitions[feature_num]

    def labels_feature_00(self, row):
        """Generate training labels for feature 00

        Feature 00:
            The presence of a blueish background color
            on the upper wing.
            True for subspecies: [1, 2, 13]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 00 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=0)

    def labels_feature_01(self, row):
        """Generate training labels for feature 01

        Feature 01:
            Inner orange pattern close to the body flowing outwards
            to about 1/3 of the wing.
            True for subspecies: [0, 5, 6, 8, 12]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 01 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=1)

    def labels_feature_02(self, row):
        """Generate training labels for feature 02

        Feature 02:
            Outer white circle speckles on the upper wing.
            Light/white points roughly aligned in the shape of
            a circle with larger opening on the bottom of the
            circle.
            True for subspecies: [0, 5]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 02 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=2)

    def labels_feature_03(self, row):
        """Generate training labels for feature 03

        Feature 03:
            Pink almost vertical line going from wing top to roughly 3/4
            down the wing. This stripe is located a little further out
            than half of the wing span.
            True for subspecies: [2]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 03 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=3)

    def labels_feature_04(self, row):
        """Generate training labels for feature 04

        Feature 04:
            Red diagonal line roughly center of wing.
            The bottom part flows further out while the top
            is closer to the body.
            True for subspecies: [3]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 04 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=4)

    def labels_feature_05(self, row):
        """Generate training labels for feature 05

        Feature 05:
            Thick orange/red diagonal line that flows outwards
            on the wing if going from top to bottom. The line
            is located roughly in the middle of the wing.
            True for subspecies: [4]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 05 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=5)

    def labels_feature_06(self, row):
        """Generate training labels for feature 06

        Feature 06:
            Orange diagonal line that flows outwards
            on the wing if going from top to bottom. The line
            is located roughly in the middle of the wing.
            In contrast to feature 05, this line is thinner.
            True for subspecies: [7, 10, 13]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 06 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=6)

    def labels_feature_07(self, row):
        """Generate training labels for feature 07

        Feature 07:
            Light/white ovalish shape in the upper, outer
            wing tip.
            True for subspecies: [6, 9]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 07 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=7)

    def labels_feature_08(self, row):
        """Generate training labels for feature 08

        Feature 08:
            Diagonal collection of white oval-like shapes a
            little further out than halway on the wing.
            Diagonally flows outward when going from top to bottom
            on the wing. Only goes to roughly half of the wing
            height. Sometimes almost vertical.
            True for subspecies: [8]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 08 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=8)

    def labels_feature_09(self, row):
        """Generate training labels for feature 09

        Feature 09:
            Inner collection of 1-2 dominant blobs.
            Often the top blob is orange and the lower one
            is white, sometimes both are white.
            These are located in the center of the wing.
            True for subspecies: [9]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 09 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=9)

    def labels_feature_10(self, row):
        """Generate training labels for feature 10

        Feature 10:
            A mostly red diagonal, zig-zagging line that goes
            down (and little outwards) on the wing from
            top to bottom. This line is more zagged than
            the other diagonal lines in features 04, 05, 06.
            This line has sharper edges and corners.
            Two distinct sharp peaks are visible on the lower
            end of the stripe
            True for subspecies: [11]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 10 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=10)

    def labels_feature_11(self, row):
        """Generate training labels for feature 11

        Feature 11:
            A dense outflow of ~8 blobs/ovals that are
            mostly in the center of the wing. These
            blobs are distributed in 2 lazers. The first
            of which contains 1-2 blobs. The second layer
            of blobs is aligned radially outwards right
            after the first layer.
            True for subspecies: [12]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 11 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=11)


class LowerWingDataHandler(SegmentDataHandler):
    def __init__(
        self,
        meta_data_path,
        data_dir_lower,
        image_processor,
        skip_hybrid=True,
        test_split=0.05,
        seed=42,
    ):
        super().__init__(
            meta_data_path=meta_data_path,
            data_dir_upper="dummy_non_existing",
            data_dir_lower=data_dir_lower,
            data_dir_noise="dummy_non_existing",
            skip_hybrid=skip_hybrid,
            image_processor=image_processor,
            test_split=test_split,
            seed=seed,
        )

        self.set_feature_definitions()

    def set_feature_definitions(self):
        self.feature_definitions = {
            0: [1, 3, 4, 10, 11],
            1: [5, 6, 8, 12],
            2: [2],
            3: [1, 2, 13],
        }

    def labels_feature(self, row, feature_num):
        """Generate training labels

        Parameters
        ----------
        row : pd.Series
            Meta data of the image
        feature_num : int
            The feature number to compute the label for.

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature is present, False (0) otherwise.
        """
        return row["subspecies"] in self.feature_definitions[feature_num]

    def labels_feature_00(self, row):
        """Generate training labels for feature 00

        Feature 00:
            The presence of a white diagonal line that goes
            upwards when going to the outer edge of the lower
            wing. Line is mostly horizontal with only a slight
            angle.
            True for subspecies: [1, 3, 4, 10, 11]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 00 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=0)

    def labels_feature_01(self, row):
        """Generate training labels for feature 01

        Feature 01:
            The presence of an orange stripe pattern on the
            lower wind that has similarities in shape to the
            skeleton of a human hand.
            True for subspecies: [5, 6, 8, 12]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 01 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=1)

    def labels_feature_02(self, row):
        """Generate training labels for feature 02

        Feature 02:
            The presence of a white/blue pattern on the bottom
            edge of the lower wing.
            True for subspecies: [2]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 02 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=2)

    def labels_feature_03(self, row):
        """Generate training labels for feature 03

        Feature 03:
            The presence of a widespread blue color
            on the lower wing. This can be strong or
            barely visible.
            True for subspecies: [1, 2, 13]

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature 03 is present, False (0) otherwise.
        """
        return self.labels_feature(row=row, feature_num=3)


class FeatureDataHandler(SegmentDataHandler):
    """Data Generator to load features"""

    def __init__(
        self,
        data_dir_features,
        num_features=12,
        wing_type="upper",
    ):
        if wing_type not in ["upper", "lower"]:
            raise ValueError(f"Unknown wing type: {wing_type}")

        self.processes = []
        self.num_features = num_features
        self.wing_type = wing_type
        self.feature_files = {}
        self.n_samples_features = {}
        for feature_num in range(self.num_features):
            feature_glob = os.path.join(
                data_dir_features,
                "*",  # subspecies directory
                f"{self.wing_type}",
                f"{self.wing_type}_{feature_num:02d}",
                "*.jpg",
            )
            self.feature_files[feature_num] = sorted(glob(feature_glob))
            self.n_samples_features[feature_num] = len(
                self.feature_files[feature_num]
            )

    def load_feature(self, feature_num, idx):
        """Load Feature image

        Parameters
        ----------
        feature_num : int
            The feature number to load.
        idx : int
            The index of the feature to load.

        Returns
        -------
        img : PIL.Image
            The loaded image
        """
        return Image.open(self.feature_files[feature_num][idx])

    def labels_feature(self, row, feature_num):
        """Generate training labels

        Parameters
        ----------
        row : pd.Series
            Meta data of the image
        feature_num : int
            The feature number to compute the label for.

        Returns
        -------
        label : int
            The label of the image.
            True (1) if feature is present, False (0) otherwise.
        """
        return row[f"{self.wing_type}_{feature_num:02d}"]

    def __call__(
        self,
        add_features=True,
        max_features=3,
        mask_only=False,
        grayscale=False,
        seed=None,
        training=True,
        sample_weights=None,
        apply_augmentations=True,
    ):
        """Load a random image and augment it

        Parameters
        ----------
        add_features : bool
            If True, synthetic images will be created with
            features overlaid on the base image.
        max_features : int
            The maximum number of features to add on top of image.
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        seed : int
            Seed for random number generator
        training : bool
            If True, sample from the training set.
            Otherwise, sample from the test set.
        sample_weights : np.ndarray
            Weights for sampling.
            If None, use uniform sampling.
        apply_augmentations : bool
            If True, apply augmentations.

        Returns
        -------
        img_aug : np.ndarray
            The augmented image
        row : pd.Series
            The meta data of the loaded image
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        # sample random image
        if training:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train)
            else:
                index = rng.choice(
                    self.indices[: self.n_samples_train],
                    p=sample_weights[: self.n_samples_train],
                )
        else:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train, self.n_samples)
            else:
                index = rng.choice(
                    self.indices[self.n_samples_train :],
                    p=sample_weights[self.n_samples_train :],
                )
        img, row = self.load_data(index)

        # augment image
        img, _ = self.image_processor.augment_image(
            img,
            mask_only=mask_only,
            grayscale=grayscale,
            apply_augmentations=apply_augmentations,
        )

        # choose which data generation method to use
        method = rng.choice([0, 1, 2])

        # keep original image
        if method == 0:
            for feature_idx in range(self.num_features):
                if row["subspecies"] in self.feature_definitions[feature_idx]:
                    row[f"{self.wing_type}_{feature_idx:02d}"] = True
                else:
                    row[f"{self.wing_type}_{feature_idx:02d}"] = False

        # create synthetic image by adding features
        # 1: on cleaned image
        # 2: on original image
        elif method in [1, 2]:
            # clean original image first
            if method == 1:
                # extract features
                base_features = data_utils.extract_features(img, tolerance=70)[
                    :max_features
                ]

                # clean base image
                img_clean = data_utils.remove_features(img, base_features)
            else:
                img_clean = img

            # select random features to add
            n_features = rng.integers(0, max_features + 1)
            feature_indices = rng.choice(
                np.arange(self.num_features),
                n_features,
                replace=False,
            )

            # remove features for which no samples exist
            feature_indices = [
                idx
                for idx in feature_indices
                if self.n_samples_features[idx] > 0
            ]
            print("Adding:", feature_indices)

            # add features
            img_features = np.zeros_like(img_clean)
            for feature_idx in feature_indices:
                # load random feature
                feature = np.asarray(
                    self.load_feature(
                        feature_num=feature_idx,
                        idx=rng.integers(
                            0, self.n_samples_features[feature_idx]
                        ),
                    )
                )

                # align images
                _, feature = data_utils.align_images(img, feature)

                mask = np.any(feature > 10, axis=-1)

                # 0: replace, 1: max, 2: weighted
                overlay_method = rng.integers(0, 2)
                if overlay_method == 0:
                    img_features[mask] = feature[mask]
                elif overlay_method == 1:
                    img_features = np.maximum(img_features, feature)
                elif overlay_method == 2:
                    alpha = rng.uniform(0.2, 0.8)
                    img_features = cv2.addWeighted(
                        img_features, alpha, feature, 1 - alpha, 0
                    )

            mask = np.any(img_features > 10, axis=-1)
            img_clean[mask] = img_features[mask]
            img = img_clean

            # create labels
            for feature_idx in range(self.num_features):
                if feature_idx in feature_indices:
                    row[f"{self.wing_type}_{feature_idx:02d}"] = True
                elif (
                    method == 2
                    and row["subspecies"]
                    in self.feature_definitions[feature_idx]
                ):
                    row[f"{self.wing_type}_{feature_idx:02d}"] = True
                else:
                    row[f"{self.wing_type}_{feature_idx:02d}"] = False

        else:
            raise ValueError(method)

        # augment image
        return img, row
        # img_aug, _ = self.image_processor.augment_image(
        #     img,
        #     mask_only=mask_only,
        #     grayscale=grayscale,
        #     apply_augmentations=apply_augmentations,
        # )
        # return img_aug, row


class UpperFeatureWingDataHandler(FeatureDataHandler, UpperWingDataHandler):
    """Data Generator for Upper Wing (with synthetic data)"""

    def __init__(
        self,
        meta_data_path,
        data_dir_upper,
        data_dir_features,
        image_processor,
        skip_hybrid=True,
        test_split=0.05,
        seed=42,
    ):
        UpperWingDataHandler.__init__(
            self=self,
            meta_data_path=meta_data_path,
            data_dir_upper=data_dir_upper,
            image_processor=image_processor,
            skip_hybrid=skip_hybrid,
            test_split=test_split,
            seed=seed,
        )
        FeatureDataHandler.__init__(
            self=self,
            data_dir_features=data_dir_features,
            num_features=12,
            wing_type="upper",
        )


class LowerFeatureWingDataHandler(FeatureDataHandler, LowerWingDataHandler):
    """Data Generator for Lower Wing (with synthetic data)"""

    def __init__(
        self,
        meta_data_path,
        data_dir_lower,
        data_dir_features,
        image_processor,
        skip_hybrid=True,
        test_split=0.05,
        seed=42,
    ):
        LowerWingDataHandler.__init__(
            self=self,
            meta_data_path=meta_data_path,
            data_dir_lower=data_dir_lower,
            image_processor=image_processor,
            skip_hybrid=skip_hybrid,
            test_split=test_split,
            seed=seed,
        )
        FeatureDataHandler.__init__(
            self=self,
            data_dir_features=data_dir_features,
            num_features=4,
            wing_type="lower",
        )


class SegmentationDataHandler:
    """Data Handler for Image Segmentation Training"""

    def __init__(
        self,
        meta_data_path,
        data_dir,
        image_processor,
        skip_hybrid=True,
        test_split=0.05,
        seed=42,
        reduction_factor=1,
    ):
        self.processes = []
        self.rng = np.random.default_rng(seed)

        self.test_split = test_split
        self.image_processor = image_processor
        self.skip_hybrid = skip_hybrid
        self.reduction_factor = reduction_factor

        self.data_dir = os.path.abspath(data_dir)
        self.df_meta = pd.read_csv(meta_data_path)

        if self.skip_hybrid:
            self.df_meta = self.df_meta[self.df_meta["subspecies"].notna()]

        # randomize order
        self.df_meta = self.df_meta.sample(frac=1, random_state=self.rng)
        self.n_samples = len(self.df_meta)
        self.n_samples_train = int(self.n_samples * (1 - self.test_split))
        self.n_samples_test = self.n_samples - self.n_samples_train

        self.indices = np.arange(self.n_samples)

    def load_data(self, index):
        """Load image and meta data

        Parameters
        ----------
        index : int
            Index of the sample to load

        Returns
        -------
        img : PIL.Image
            The loaded image
        row : pd.Series
            The meta data of the loaded image
        """
        row = pd.Series(self.df_meta.iloc[index])
        img_path = os.path.join(
            self.data_dir,
            f"{row['filename'][:-4]}_image.jpg",
        )
        img = Image.open(img_path)
        return img, row

    def load_by_name(self, name):
        """Load image and meta data by name

        Parameters
        ----------
        name : str
            Name of the sample to load

        Returns
        -------
        img : PIL.Image
            The loaded image
        row : pd.Series
            The meta data of the loaded image
        """
        row = self.df_meta[self.df_meta["filename"] == name].iloc[0]
        img_path = os.path.join(
            self.data_dir,
            f"{row['filename'][:-4]}_image.jpg",
        )
        img = Image.open(img_path)
        return img, row

    def segmentation_labels(self, row):
        """Generate training labels for segmentation

        Parameters
        ----------
        row : pd.Series
            Meta data of the image

        Returns
        -------
        label : np.ndarray
            The label of each pixel of the image.
            Shape: (height, width, num_classes)
        """
        img_path = os.path.join(
            self.data_dir,
            f"{row['filename'][:-4]}_mask.jpg",
        )
        img = Image.open(img_path)
        img = np.argmax(img, axis=-1)
        return img

    def __call__(
        self,
        mask_only=False,
        grayscale=False,
        seed=None,
        training=True,
        sample_weights=None,
        apply_augmentations=True,
    ):
        """Load a random image and augment it

        Parameters
        ----------
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        seed : int
            Seed for random number generator
        training : bool
            If True, sample from the training set.
            Otherwise, sample from the test set.
        sample_weights : np.ndarray
            Weights for sampling.
            If None, use uniform sampling.
        apply_augmentations : bool
            If True, apply augmentations.

        Returns
        -------
        img_aug : np.ndarray
            The augmented image
        mask_aug : np.ndarray
            The augmented mask
        row : pd.Series
            The meta data of the loaded image
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        # sample random image
        if training:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train)
            else:
                index = rng.choice(
                    self.indices[: self.n_samples_train],
                    p=sample_weights[: self.n_samples_train],
                )
        else:
            if sample_weights is None:
                index = rng.integers(self.n_samples_train, self.n_samples)
            else:
                index = rng.choice(
                    self.indices[self.n_samples_train :],
                    p=sample_weights[self.n_samples_train :],
                )
        img, row = self.load_data(index)
        img = np.asarray(img)
        mask = self.segmentation_labels(row)

        if self.reduction_factor > 1:
            new_size = np.array(mask.shape) // self.reduction_factor
            result = A.Resize(*new_size)(image=img, mask=mask)
            img = result["image"]
            mask = result["mask"]

        # augment image
        img_aug, mask_aug = self.image_processor.augment_image(
            img,
            mask=mask,
            mask_only=mask_only,
            grayscale=grayscale,
            apply_augmentations=apply_augmentations,
        )
        return img_aug, mask_aug, row

    def get_generator(
        self,
        batch_size=32,
        queue_size=32,
        mask_only=False,
        grayscale=False,
        training=True,
        balanced_loading=False,
        n_jobs=1,
    ):
        """Get a data generator

        Parameters
        ----------
        batch_size : int
            Batch size.
        queue_size : int
            Size of the queue.
        labels_func_name : str
            The name of the function to generate labels.
        mask_only : bool
            If True, only return the mask.
        grayscale : bool
            If True, convert the image to grayscale.
        training : bool
            If True, sample from the training set.
            Otherwise, sample from the test set.
        balanced_loading : bool
            If True, load samples balanced.
        n_jobs : int
            Number of processes to use

        Returns
        -------
        generator : function
            A function that generates augmented images
        """
        if balanced_loading:
            raise NotImplementedError

        queue = mp.Manager().Queue(maxsize=queue_size)

        def worker(seed):
            rng = np.random.default_rng(seed)
            while True:
                img_aug, mask_aug, _ = self(
                    mask_only=mask_only,
                    grayscale=grayscale,
                    training=training,
                    sample_weights=None,
                    seed=rng.integers(2**32),
                )
                queue.put((img_aug, mask_aug))

        for i in range(n_jobs):
            process = mp.Process(target=worker, args=(i,))
            process.start()
            self.processes.append(process)

        def generator():
            while True:
                batch_images = []
                batch_labels = []
                while len(batch_images) < batch_size:
                    img_aug, mask_aug = queue.get()
                    batch_images.append(img_aug)
                    batch_labels.append(mask_aug)
                images = np.stack(batch_images, axis=0)
                labels = np.stack(batch_labels, axis=0)
                yield images, labels

        return generator()

    def __del__(self):
        for process in self.processes:
            process.terminate()


class DataHandler:
    def __init__(self, meta_data_path, data_dir):
        """Initialize the data handler

        Parameters
        ----------
        meta_data_path : str
            Path to the meta data file
        data_dir : str
            Path to the directory containing the images.
        """
        self.meta_data = pd.read_csv(meta_data_path)
        self.data_dir = data_dir

        # load meta data
        self.df_meta = pd.read_csv(meta_data_path)

        self.n_samples = len(self.df_meta)

    def load_by_name(self, name):
        """Load image and meta data by name

        Parameters
        ----------
        name : str
            Name of the sample to load

        Returns
        -------
        img : PIL.Image
            The loaded image
        row : pd.Series
            The meta data of the loaded image
        """
        if not name.endswith(".jpg"):
            name += ".jpg"
        row = self.df_meta[self.df_meta["filename"] == name].iloc[0]
        img_path = os.path.join(
            self.data_dir, row["hybrid_stat"], row["filename"]
        )
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        return img, row

    def load_data(self, index):
        """Load image and meta data

        Parameters
        ----------
        index : int
            Index of the sample to load

        Returns
        -------
        img : PIL.Image
            The loaded image
        row : pd.Series
            The meta data of the loaded image
        """
        row = self.df_meta.iloc[index]
        img_path = os.path.join(
            self.data_dir, row["hybrid_stat"], row["filename"]
        )
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        return img, row
