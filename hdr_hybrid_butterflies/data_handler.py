import os
from glob import glob
import numpy as np
import pandas as pd
import imagesize
from PIL import Image
import cv2
import albumentations as A
import multiprocessing as mp

from hdr_hybrid_butterflies.dino_sam import dino_sam


class ImageProcessor:
    def __init__(
        self,
        detector_id="IDEA-Research/grounding-dino-tiny",
        segmenter_id="facebook/sam-vit-base",
        labels=["wings."],
        threshold=0.2,
        output_dim=(256, 256),
    ):
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.labels = labels
        self.threshold = threshold
        self.output_dim = output_dim

        # Define augmentations
        p0 = 0.75
        p1 = 0.25
        p2 = 0.1
        self.augmentations = A.Compose(
            [
                A.AdditiveNoise(noise_type="uniform", p=p1),
                A.Blur(blur_limit=3, p=p2),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=p1,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=p1
                ),
                A.CLAHE(clip_limit=2.0, p=p2),
                A.RGBShift(
                    r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=p1
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
            ]
        )

        self.resize = A.Resize(*output_dim)

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

    def augment_image(self, image, mask_only=False):
        """Augment an image

        Parameters
        ----------
        image : cv2.Image | PIL.Image | np.ndarray
            The image to augment.
        mask_only : bool
            If True, only return the mask.

        Returns
        -------
        np.ndarray
            The augmented image.
        """
        image = np.array(image)
        mask = np.asarray(np.any(image != 0, axis=-1), dtype=np.uint8)
        augmentation = self.augmentations(
            image=image,
            mask=mask,
        )

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
        data_dir_upper,
        data_dir_lower,
        data_dir_noise,
        image_processor,
        test_split=0.2,
        seed=42,
    ):
        """Initialize the data handler

        Parameters
        ----------
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

        # create meta dataframe
        self.df_meta = {
            "filename": [],
            "label": [],
            "score": [],
            "width": [],
            "height": [],
            "ratio": [],
        }
        for data_dir, label in zip(
            [data_dir_upper, data_dir_lower, data_dir_noise],
            ["upper", "lower", "noise"],
        ):
            file_list = sorted(glob(os.path.join(data_dir, "*.jpg")))
            for filename in file_list:
                score = float(filename.split("_")[-1][1:-4])
                width, height = imagesize.get(filename)
                ratio = width / height
                score = float(filename.split("_")[-1][1:-4])

                self.df_meta["filename"].append(os.path.basename(filename))
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

    def __call__(self, mask_only=False, seed=None, training=True):
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
            index = rng.integers(self.n_samples_train)
        else:
            index = rng.integers(self.n_samples_train, self.n_samples)
        img, row = self.load_data(index)

        # augment image
        img_aug = self.image_processor.augment_image(
            img,
            mask_only=mask_only,
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

    def get_generator(
        self,
        batch_size=32,
        queue_size=32,
        labels_func_name="segmentation_labels",
        mask_only=False,
        training=True,
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
        n_jobs : int
            Number of processes to use

        Returns
        -------
        generator : function
            A function that generates augmented images
        """
        queue = mp.Manager().Queue(maxsize=queue_size)

        label_creater = getattr(self, labels_func_name)

        def worker(seed):
            rng = np.random.default_rng(seed)
            while True:
                img_aug, row = self(
                    mask_only=mask_only,
                    training=training,
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
        return img, row
