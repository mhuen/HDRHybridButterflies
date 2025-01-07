"""
Submission model file.
The ingestion program will call `predict` to get a prediction for each
test image and then save the predictions for scoring.
The following two methods are required:
- predict: uses the model to perform predictions.
- load: reloads the model.
"""
import os
import numpy as np

from hdr_hybrid_butterflies.data_handler import ImageProcessor
from hdr_hybrid_butterflies.model import CNNClasifier


class Model:
    def __init__(self):
        # model will be called from the load() method
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.models_dir = os.path.join(self.dir, "models")

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
        self.image_processor = ImageProcessor(
            segment_classifier=self.segment_classifier,
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

        lower_segments, upper_segments = self.image_processor(datapoint)

        probabilities = self.hybrid_classifier.probabilities(upper_segments)

        result = np.mean(probabilities, axis=0)[1]
        print(f"Hybrid butterfly score: {result}")
        return result
