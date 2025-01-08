import os
import click
import numpy as np
import tensorflow as tf
import socket

from hdr_hybrid_butterflies.model import CNNClasifier, WingCNNClasifier
from hdr_hybrid_butterflies.data_handler import (
    UpperWingDataHandler,
    LowerWingDataHandler,
    SegmentDataHandler,
    WingSegmentDataHandler,
    ImageProcessor,
)


@click.command()
@click.argument(
    "classifier_type",
    type=str,
    required=1,
)
@click.option(
    "-f",
    "--feature_number",
    type=int,
    default=0,
)
@click.option(
    "--load_weights",
    is_flag=True,
    help="Load weights from previous training run",
    default=False,
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train the model",
)
@click.option(
    "-s",
    "--steps_per_epoch",
    type=int,
    default=100,
    help="Number of steps per epoch",
)
@click.option(
    "-b",
    "--batch_size",
    type=int,
    default=16,
    help="Batch size",
)
@click.option(
    "--balanced_loading",
    is_flag=True,
    help="Load data balanced",
    default=False,
)
@click.option(
    "-j",
    "--n_jobs",
    type=int,
    default=12,
    help="Number of jobs for data generator",
)
def main(
    classifier_type,
    feature_number,
    load_weights,
    epochs,
    steps_per_epoch,
    batch_size,
    balanced_loading,
    n_jobs,
):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if socket.gethostname() == "jupyter-mhuennefeld":
        data_dir = "../../data/images/"
        meta_data_path = "../../data/butterfly_anomaly_train.csv"
        model_dir = "../../data/models/"
    else:
        data_dir = "../../data/images/"
        meta_data_path = "../../data/butterfly_anomaly_train.csv"
        model_dir = "../../data/models/"

    segment_training_dir = os.path.join(data_dir, "segment_classier_training")
    for dir_path in [model_dir, segment_training_dir]:
        if not os.path.exists(dir_path):
            print("Creating directory:", dir_path)
            os.makedirs(dir_path)

    image_processor = ImageProcessor()

    print(
        f"Training {classifier_type} classifier for feature {feature_number}"
    )

    if classifier_type == "upper":
        model_class = CNNClasifier
        mask_only = False
        num_classes = 2
        labels_func_name = f"labels_feature_{feature_number:02d}"
        model_name = f"upper_wing_feature_{feature_number:02d}"
        checkpoint_path = os.path.join(
            model_dir, f"upper_model_{feature_number:02d}", "model.weights.h5"
        )

        data_handler = UpperWingDataHandler(
            meta_data_path=meta_data_path,
            data_dir_upper=os.path.join(
                segment_training_dir, "manual", "upper_wing_manual"
            ),
            image_processor=image_processor,
        )
    elif classifier_type == "lower":
        model_class = CNNClasifier
        mask_only = False
        num_classes = 2
        labels_func_name = f"labels_feature_{feature_number:02d}"
        model_name = f"lower_wing_feature_{feature_number:02d}"
        checkpoint_path = os.path.join(
            model_dir, f"lower_model_{feature_number:02d}", "model.weights.h5"
        )

        data_handler = LowerWingDataHandler(
            meta_data_path=meta_data_path,
            data_dir_lower=os.path.join(
                segment_training_dir, "manual", "lower_wing_manual"
            ),
            image_processor=image_processor,
        )

    elif classifier_type == "segment":
        model_class = CNNClasifier
        mask_only = True
        num_classes = 3
        labels_func_name = "segmentation_labels"
        model_name = "segment_model"
        checkpoint_path = os.path.join(
            model_dir, "segment_model", "model.weights.h5"
        )

        data_handler = SegmentDataHandler(
            meta_data_path=meta_data_path,
            data_dir_upper=os.path.join(
                segment_training_dir, "manual", "upper_wing_manual"
            ),
            data_dir_lower=os.path.join(
                segment_training_dir, "manual", "lower_wing_manual"
            ),
            data_dir_noise=os.path.join(
                segment_training_dir, "manual", "noise_manual"
            ),
            image_processor=image_processor,
        )
    elif classifier_type == "wing_subspecies":
        model_class = WingCNNClasifier
        mask_only = False
        num_classes = 2
        labels_func_name = f"label_subspecies_{feature_number:02d}"
        model_name = "wing_subspecies_model"
        checkpoint_path = os.path.join(
            model_dir, "wing_subspecies_model", "model.weights.h5"
        )

        data_handler = WingSegmentDataHandler(
            meta_data_path=meta_data_path,
            data_dir_upper=os.path.join(
                segment_training_dir, "manual", "upper_wing_manual"
            ),
            data_dir_lower=os.path.join(
                segment_training_dir, "manual", "lower_wing_manual"
            ),
            image_processor=image_processor,
        )

    elif classifier_type == "signal_hybrid":
        model_class = CNNClasifier
        mask_only = False
        num_classes = 2
        labels_func_name = "hybrid_labels"
        model_name = "signal_hybrid_model"
        checkpoint_path = os.path.join(
            model_dir, "signal_hybrid_model", "model.weights.h5"
        )

        data_handler = UpperWingDataHandler(
            meta_data_path=meta_data_path,
            data_dir_upper=os.path.join(
                segment_training_dir, "manual", "upper_wing_manual"
            ),
            image_processor=image_processor,
            skip_hybrid=False,
        )

    else:
        raise ValueError(f"Segment type {classifier_type} not supported")

    # print out the distribution of subspecies in the training data
    result = np.unique(
        data_handler.df_meta["subspecies"].iloc[
            : data_handler.n_samples_train
        ],
        return_counts=True,
    )
    print("Subspecies distribution in training data:")
    for sub, count in zip(*result):
        print(f"Subspecies: {sub} | Count: {count}")

    # Create data iterator
    generator_train = data_handler.get_generator(
        batch_size=batch_size,
        queue_size=10000,
        n_jobs=n_jobs,
        mask_only=mask_only,
        training=True,
        balanced_loading=balanced_loading,
        labels_func_name=labels_func_name,
    )

    generator_test = data_handler.get_generator(
        batch_size=batch_size,
        queue_size=320,
        n_jobs=1,
        mask_only=mask_only,
        training=False,
        labels_func_name=labels_func_name,
    )

    # Create model
    model = model_class(
        image_size=image_processor.output_dim,
        num_classes=num_classes,
        name=model_name,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    if load_weights:
        model.load_weights(checkpoint_path)

    # Create the ModelCheckpoint callback
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="loss",  # 'val_loss'
        mode="min",
        save_best_only=False,
        save_freq="epoch",  # Save every epoch
        verbose=1,
    )

    model.fit(
        generator_train,
        validation_data=generator_test,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=5,
        callbacks=[model_checkpoint_callback],
    )


if __name__ == "__main__":
    main()
