"""
Submission model file.
The ingestion program will call `predict` to get a prediction for each
test image and then save the predictions for scoring.
The following two methods are required:
- predict: uses the model to perform predictions.
- load: reloads the model.
"""
import os
import timeit
import numpy as np

from hdr_hybrid_butterflies.data_handler import ImageProcessor
from hdr_hybrid_butterflies.model import CNNClasifier, CNNSegmenter
from hdf_hybrid_butterflies.config import CNN_SEGMENTER_IMAGE_SIZE


class Model:
    def __init__(self):
        # model will be called from the load() method
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.models_dir = os.path.join(self.dir, "models")
        self.counter = 0
        self.t_start = timeit.default_timer()

    def load(self):
        self.segment_classifier = CNNClasifier(
            image_size=ImageProcessor().output_dim,
            num_classes=3,
            verbose=False,
        )
        self.segment_classifier.load_weights(
            os.path.join(
                self.models_dir,
                "segment_model",
                "model.weights.h5",
            )
        )

        self.cnn_segmenter = CNNSegmenter(
            num_classes=3,
            image_size=CNN_SEGMENTER_IMAGE_SIZE,
        )
        self.cnn_segmenter.load_weights(
            os.path.join(
                self.models_dir,
                "segmentation_model",
                "model.weights.h5",
            )
        )

        self.cnn_segmenter_processor = ImageProcessor(
            segment_classifier=self.segment_classifier,
            p_erase=0.0,
            padding_size=0,
            output_dim=CNN_SEGMENTER_IMAGE_SIZE,
        )

        self.image_processor = ImageProcessor(
            segment_classifier=self.segment_classifier,
            cnn_segmenter=self.cnn_segmenter,
            cnn_segmenter_processor=self.cnn_segmenter_processor,
        )

        self.hybrid_classifier = CNNClasifier(
            image_size=self.image_processor.output_dim,
            num_classes=2,
            verbose=False,
        )
        self.hybrid_classifier.load_weights(
            os.path.join(
                self.models_dir,
                "signal_hybrid_model",
                "model.weights.h5",
            )
        )

    def predict(self, datapoint):
        """Predict if the image is a hybrid butterfly.

        Parameters
        ----------
        datapoint : PIL.Image
            Image to predict.

        Returns
        -------
        float
            Probability that the image is a hybrid butterfly.
        """
        self.counter += 1
        print(f"Image {self.counter} processed.")
        t_start = timeit.default_timer()
        print(f"Current time: {t_start - self.t_start}")

        # abort if we are running out of time
        if t_start - self.t_start > 450:
            return 0.5

        try:
            lower_segments, upper_segments = self.image_processor(
                datapoint,
                via_cnn=True,
            )
            print(f"Image processing time: {timeit.default_timer() - t_start}")
            probabilities = self.hybrid_classifier.probabilities(
                upper_segments
            )
        except Exception as e:
            print(f"Error: {e}")
            return 0.5

        result = np.mean(probabilities, axis=0)[1]
        print(f"Time take: {timeit.default_timer() - t_start}")
        print(f"Hybrid butterfly score: {result}")
        return result
