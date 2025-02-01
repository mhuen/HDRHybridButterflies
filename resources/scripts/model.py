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
import pickle
import numpy as np
import tensorflow as tf

from hdr_hybrid_butterflies.data_handler import ImageProcessor
from hdr_hybrid_butterflies.data_utils import compute_anomaly
from hdr_hybrid_butterflies.model import (
    CNNClasifier,
    CNNSegmenter,
    WingCNNClasifier,
)
from hdr_hybrid_butterflies import config

print(
    "[Tensorflow] Num GPUs Available: ",
    len(tf.config.list_physical_devices("GPU")),
)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
# if len(tf.config.list_physical_devices('GPU')) == 0:
#    raise Exception("No GPU available.")


class Model:
    def __init__(self):
        # model will be called from the load() method
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.models_dir = os.path.join(self.dir, "models")
        self.counter = 0
        self.t_start = timeit.default_timer()

        # some setting
        self.add_hybrid_stitcher = False

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
            image_size=config.CNN_SEGMENTER_IMAGE_SIZE,
            cnn_config=config.CNN_SEGMENTER_CONFIG_V01,
            verbose=False,
        )
        self.cnn_segmenter.load_weights(
            os.path.join(
                self.models_dir,
                "segmentation_model",
                "model.weights.h5",
            )
        )

        if self.add_hybrid_stitcher:
            self.hybrid_stitcher_classifier = WingCNNClasifier(
                image_size=ImageProcessor().output_dim,
                num_classes=2,
                verbose=False,
            )
            self.hybrid_stitcher_classifier.load_weights(
                os.path.join(
                    self.models_dir,
                    "hybrid_stitcher_model",
                    "model.weights.h5",
                )
            )

        self.cnn_segmenter_processor = ImageProcessor(
            segment_classifier=self.segment_classifier,
            p_erase=0.0,
            padding_size=0,
            output_dim=config.CNN_SEGMENTER_IMAGE_SIZE,
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

        # load feature classifiers
        old_preference = [0, 8, 11]
        # old_preference = [0, 11]
        self.model_dict = {}
        self.model_keys_upper = []
        self.model_keys_lower = []

        # upper wing models
        for feature_number in range(12):
            name = f"upper_feature_model_{feature_number:02d}"
            if (
                not os.path.exists(os.path.join(self.models_dir, name))
                or feature_number in old_preference
            ):
                name = f"upper_model_{feature_number:02d}"
                print(f"Falling back to model: {name}")
            self.model_keys_upper.append(name)

            self.model_dict[name] = CNNClasifier(
                image_size=self.image_processor.output_dim,
                num_classes=2,
                name=name,
                verbose=False,
            )
            self.model_dict[name].load_weights(
                os.path.join(self.models_dir, name, "model.weights.h5")
            )

        # lower wing models
        for feature_number in range(4):
            name = f"lower_feature_model_{feature_number:02d}"
            if not os.path.exists(os.path.join(self.models_dir, name)):
                name = f"lower_model_{feature_number:02d}"
                print(f"Falling back to model: {name}")
            self.model_keys_lower.append(name)

            self.model_dict[name] = CNNClasifier(
                image_size=self.image_processor.output_dim,
                num_classes=2,
                name=name,
                verbose=False,
            )
            self.model_dict[name].load_weights(
                os.path.join(self.models_dir, name, "model.weights.h5")
            )

        self.model_keys = self.model_keys_upper + self.model_keys_lower
        self.n_features = len(self.model_keys_upper) + len(
            self.model_keys_lower
        )

        # build anomaly model based on features
        with open(
            os.path.join(self.models_dir, "isolation_forest_features.pkl"),
            "rb",
        ) as f:
            self.clf, model_keys = pickle.load(f)

        # fmt: off
        self.truth_matrix = np.array([
            [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
            [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        ])
        # fmt: on

        assert model_keys == self.model_keys, (model_keys, self.model_keys)

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
        print(f"Processing image {self.counter}.")
        t_start = timeit.default_timer()
        print(f"  Current time: {t_start - self.t_start}")

        # abort if we are running out of time
        # if t_start - self.t_start > 550:
        #    return 0.5

        try:
            results = []

            # ---------------------------------------------------
            # extract the upper and lower segments from the image
            # ---------------------------------------------------
            lower_segments, upper_segments = self.image_processor(
                datapoint,
                via_cnn=True,
            )
            assert len(upper_segments) > 0, len(upper_segments)
            assert len(lower_segments) > 0, len(lower_segments)

            t_processing = timeit.default_timer()
            print(f"  Image processing time: {t_processing - t_start}")

            # --------------------------------
            # run the signal hybrid classifier
            # --------------------------------
            probabilities = self.hybrid_classifier.probabilities(
                upper_segments
            )
            prob_hybrid = np.mean(probabilities, axis=0)[1]
            results.append(prob_hybrid)

            t_hybrid_classifier = timeit.default_timer()
            print(f"  Prediction time: {t_hybrid_classifier - t_processing}")

            # -----------------------------------------
            # run the feature model anomaly classifiers
            # -----------------------------------------
            predictions = np.zeros((self.n_features))
            idx = 0
            for model_name in self.model_keys_upper:
                model = self.model_dict[model_name]
                predictions[idx] = np.max(
                    model.probabilities(upper_segments)[:, 1], axis=0
                )
                idx += 1

            for model_name in self.model_keys_lower:
                model = self.model_dict[model_name]
                predictions[idx] = np.max(
                    model.probabilities(lower_segments)[:, 1], axis=0
                )
                idx += 1
            predictions = predictions[None, :]

            # anomaly_score = -self.clf.score_samples(predictions)[0]
            anomaly_score = compute_anomaly(predictions, self.truth_matrix)[0]

            # calibrate the anomaly score
            anomaly_score = anomaly_score**5

            results.append(anomaly_score)
            t_anomaly_classifier = timeit.default_timer()
            print(
                "  Anomaly prediction time: "
                f"{t_anomaly_classifier - t_hybrid_classifier}"
            )

            # -------------------------
            # Add hybrid stitcher model
            # -------------------------
            if self.add_hybrid_stitcher:
                zeros = np.zeros_like(upper_segments)
                if len(upper_segments) == 1:
                    upper_segments = np.concatenate(
                        [upper_segments, zeros], axis=0
                    )
                if len(lower_segments) == 1:
                    lower_segments = np.concatenate(
                        [lower_segments, zeros], axis=0
                    )

                assert len(upper_segments) == 2, len(upper_segments)
                assert len(lower_segments) == 2, len(lower_segments)

                wing = np.concatenate(
                    [upper_segments, lower_segments], axis=0
                )[None]

                assert wing.shape == (1, 4, 256, 256, 3), wing.shape
                probabilities_stitcher = (
                    self.hybrid_stitcher_classifier.probabilities(wing)
                )
                prob_stitcher = np.mean(probabilities_stitcher, axis=0)[1]
                results.append(prob_stitcher)
                t_stitcher = timeit.default_timer()
                print(
                    "  Stitcher prediction time: "
                    f"{t_stitcher - t_hybrid_classifier}"
                )
                print(f"  --> Hybrid stitcher score: {prob_stitcher}")

        except Exception as e:
            print(f"  Error: {e}")
            return 0.5

        print(f"  --> Hybrid butterfly score: {prob_hybrid}")
        print(f"  --> Anomaly score: {anomaly_score}")
        print(f"  --> Time taken: {timeit.default_timer() - t_start}")
        return np.max(results)
