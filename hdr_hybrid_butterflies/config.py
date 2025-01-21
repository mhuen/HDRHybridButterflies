CNN_SEGMENTER_IMAGE_SIZE = (768, 768)


CNN_SEGMENTER_CONFIG_V01 = {
    "filter_size_list": [[7, 7]] * 20,
    "num_filters_list": [12] * 20,
    "pooling_type_list": None,
    "pooling_strides_list": [[1, 2, 2, 1]] * 20,
    "pooling_ksize_list": [[1, 2, 2, 1]] * 20,
    "strides_list": [1, 1, 1, 1],
    "dilation_rate_list": [[1, 1, 1, 1], [1, 5, 5, 1]] * 10,
    "use_dropout_list": False,
    "use_batch_normalisation_list": False,
    "activation_list": "elu",
    "use_residual_list": True,
    "method_list": "convolution",
}


CNN_SEGMENTER_CONFIG_V02 = {
    "filter_size_list": [[5, 5]] * 80,
    "num_filters_list": [8] * 80,
    "pooling_type_list": None,
    "pooling_strides_list": [[1, 2, 2, 1]] * 80,
    "pooling_ksize_list": [[1, 2, 2, 1]] * 80,
    "strides_list": [1, 1, 1, 1],
    "dilation_rate_list": [[1, 1, 1, 1], [1, 1, 1, 1]] * 40,
    "use_dropout_list": False,
    "use_batch_normalisation_list": False,
    "activation_list": "elu",
    "use_residual_list": True,
    "method_list": "convolution",
}
