from typing import Any, Tuple
import tensorflow as tf
from tfscripts import layers as tfs


class CNNClasifier(tf.keras.Model):
    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int],
        cnn_config: dict[str:Any] = {
            "filter_size_list": [[3, 3]] * 20,
            "num_filters_list": [16] * 20,
            "pooling_type_list": [
                None,
                "max",
                None,
                "max",
                None,
                None,
                "max",
                None,
                "max",
                None,
                None,
                "max",
                None,
                "max",
                None,
                None,
                "max",
                None,
                None,
                "max",
            ],
            "pooling_strides_list": [[1, 2, 2, 1]] * 20,
            "pooling_ksize_list": [[1, 2, 2, 1]] * 20,
            "strides_list": [1, 1, 1, 1],
            "dilation_rate_list": None,
            "use_dropout_list": False,
            "use_batch_normalisation_list": False,
            "activation_list": "elu",
            "use_residual_list": True,
            "method_list": "convolution",
        },
        fc_config: dict[str, Any] = {
            "fc_sizes": [64, 64, -1],
            "use_dropout_list": False,
            "activation_list": ["elu", "elu", None],
            "use_batch_normalisation_list": False,
            "use_residual_list": [True, True, True],
        },
        dtype: str = "float32",
        verbose: bool = True,
        name: str = "CNNClassifier",
    ) -> None:
        """Initialize the CNNClassifier",.

        Parameters
        ----------
        word_list_guesses : list[str]
            List of words to use as guesses for the agent.
        word_list_solutions : list[str]
            List of words used as a possible solution.
        max_guesses : int, optional
            Maximum number of guesses to make, by default 6.
        combined_config : dict[str, Any], optional
            Configuration for the combined CNN layers. These are
            passed on to the `tfscripts.layers.ConvNdLayers` class.
        fc_config : dict[str, Any], optional
            Configuration for the fully connected layers. These are
            passed on to the `tfscripts.layers.FCLayers` class.
        dtype : str, optional
            Data type to use for the model, by default "float32".
        verbose : bool, optional
            Whether to print verbose output, by default True.
        name : str, optional
            Name of the model, by default "CNNClassifier",".
        """
        tf.keras.Model.__init__(self, dtype=dtype)

        self.num_classes = num_classes
        self.image_size = image_size
        self.verbose = verbose
        self.name = name

        # cnn for combined pairs
        self.cnn = tfs.ConvNdLayers(
            input_shape=[-1, image_size[0], image_size[1], 3],
            padding_list="SAME",
            hex_zero_out_list=False,
            hex_num_rotations_list=1,
            float_precision=dtype,
            name=name + "__cnn",
            verbose=verbose,
            **cnn_config,
        )

        # update the last layer to have the correct number of classes
        fc_config["fc_sizes"][-1] = num_classes

        # create weights and layers
        self.fc_layers = tfs.FCLayers(
            input_shape=[
                -1,
                cnn_config["num_filters_list"][-1],
            ],
            weights_list=None,
            biases_list=None,
            max_out_size_list=None,
            float_precision=dtype,
            name=name + "__fc_layer",
            verbose=verbose,
            **fc_config,
        )

        # explicitly add these variables to the module
        # variables created in sub-module are not found otherwise..
        self._cnn_vars = self.cnn.variables
        self._fc_vars = self.fc_layers.variables

    def call(self, inputs: tf.Tensor, is_training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to the CNN. This is the game state.
            Shape: (batch_size, 5, n_alphabet, max_guesses*3)
        is_training : bool, optional
            Whether the model is training, by default False.

        Returns
        -------
        tf.Tensor
            Logits for the model.
            Shape: (batch_size, 5, n_alphabet)
        """

        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        inputs = tf.cast(inputs, self.dtype)
        inputs /= 255.0

        # apply cnn
        cnn_out = self.cnn(inputs, is_training=is_training)[-1]
        if is_training and self.verbose:
            print("cnn_out", cnn_out.shape)

        # flatten the output
        flattened_layer, num_features = tfs.flatten_layer(cnn_out)
        if is_training and self.verbose:
            print("flattened_layer", flattened_layer.shape)

        # apply fc layers
        logits = self.fc_layers(flattened_layer, is_training=is_training)[-1]

        return logits

    def logits2probs(self, logits: tf.Tensor) -> tf.Tensor:
        """Convert logits to probabilities using softmax.

        Parameters
        ----------
        logits : tf.Tensor
            Logits from the model.

        Returns
        -------
        tf.Tensor
            Probabilities for each class.
        """
        return tf.nn.softmax(logits, axis=-1)

    def probabilities(
        self, inputs: tf.Tensor, is_training: bool = False
    ) -> tf.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to the CNN. This is the game state.
            Shape: (batch_size, 5, n_alphabet, max_guesses*3)
        is_training : bool, optional
            Whether the model is training, by default False.

        Returns
        -------
        tf.Tensor
            Probabilities for the model.
            Shape: (batch_size, 5, n_alphabet)
        """
        logits = self.call(inputs, is_training=is_training)
        return self.logits2probs(logits)
