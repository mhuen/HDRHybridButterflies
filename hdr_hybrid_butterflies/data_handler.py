import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import albumentations as A
import multiprocessing as mp

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
        p_erase=0.5,
    ):
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.labels = labels
        self.threshold = threshold
        self.output_dim = output_dim
        self.segment_classifier = segment_classifier
        self.p_erase = p_erase

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
                A.Pad(100, p=1.0),
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
            ]
        )

        self.resize = A.Resize(*output_dim)

    def __call__(self, image, mask_only=False):
        """Process in image into a format suitable for the model

        Parameters
        ----------
        image : PIL.Image
            The image to process
        mask_only : bool
            If True, only return the mask.

        Returns
        -------
        List[np.ndarray]
            A list of lower wing segments
        List[np.ndarray]
            A list of upper wing segments
        """
        image = ImageOps.exif_transpose(image)

        # extract segments from image
        segments, scores = self._raw_segments(image)

        # transform segments for model input
        processed_segments = np.stack(
            [
                self.augment_image(
                    segment,
                    mask_only=True,
                    apply_augmentations=False,
                )
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

        # get the segments
        processed_segments = [
            self.augment_image(
                segment,
                mask_only=mask_only,
                apply_augmentations=False,
            )
            for segment in segments
        ]
        upper_segments = [processed_segments[idx] for idx in upper_list]
        lower_segments = [processed_segments[idx] for idx in lower_list]

        return np.stack(lower_segments), np.stack(upper_segments)

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
        return segments, scores

    def augment_image(self, image, mask_only=False, apply_augmentations=True):
        """Augment an image

        Parameters
        ----------
        image : cv2.Image | PIL.Image | np.ndarray
            The image to augment.
        mask_only : bool
            If True, only return the mask.
        apply_augmentations : bool
            If True, apply augmentations.
            If False, only resize the image.

        Returns
        -------
        np.ndarray
            The augmented image.
        """
        image = np.array(image)
        mask = np.asarray(np.any(image != 0, axis=-1), dtype=np.uint8)
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
        image = image * mask[:, :, None]

        # pad image to square
        square_size = max(image.shape[:2])
        pad_axis = 0 if image.shape[0] < image.shape[1] else 1
        pad_size = square_size - image.shape[pad_axis]
        pad_half = pad_size // 2

        new_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)

        if pad_axis == 0:
            new_image[pad_half : pad_half + image.shape[0], :, :] = image
        else:
            new_image[:, pad_half : pad_half + image.shape[1], :] = image

        # resize to output dim
        new_image = self.resize(image=new_image)["image"]

        if mask_only:
            mask_new_image = np.any(new_image != 0, axis=-1)
            new_image[mask_new_image] = 255

        return new_image


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
        img_aug = self.image_processor.augment_image(
            img,
            mask_only=mask_only,
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
                apply_augmentations=apply_augmentations,
            )
            for segment in segments_upper
        ]
        segments_lower = [
            self.image_processor.augment_image(
                segment,
                mask_only=mask_only,
                apply_augmentations=apply_augmentations,
            )
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
        return row["subspecies"] in self.feature_definitions[0]

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
        return row["subspecies"] in self.feature_definitions[1]

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
        return row["subspecies"] in self.feature_definitions[2]

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
        return row["subspecies"] in self.feature_definitions[3]

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
        return row["subspecies"] in self.feature_definitions[4]

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
        return row["subspecies"] in self.feature_definitions[5]

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
        return row["subspecies"] in self.feature_definitions[6]

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
        return row["subspecies"] in self.feature_definitions[7]

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
        return row["subspecies"] in self.feature_definitions[8]

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
        return row["subspecies"] in self.feature_definitions[9]

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
        return row["subspecies"] in self.feature_definitions[10]

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
        return row["subspecies"] in self.feature_definitions[11]


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

        self.feature_definitions = {
            0: [1, 3, 4, 10, 11],
            1: [5, 6, 8, 12],
            2: [2],
            3: [1, 2, 13],
        }

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
        return row["subspecies"] in self.feature_definitions[0]

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
        return row["subspecies"] in self.feature_definitions[1]

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
        return row["subspecies"] in self.feature_definitions[2]

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
        return row["subspecies"] in self.feature_definitions[3]


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
