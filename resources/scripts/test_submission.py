import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from hdr_hybrid_butterflies.data_handler import DataHandler  # noqa
from model import Model  # noqa

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def main():
    data_handler = DataHandler(
        meta_data_path=os.path.join(
            THIS_FOLDER, "butterfly_anomaly_train.csv"
        ),
        data_dir=os.path.join(THIS_FOLDER, "images"),
    )

    imgage_hybrid = data_handler.load_by_name("CAM016784")[0]
    image_non_hybrid = data_handler.load_by_name("CAM041144")[0]

    model = Model()
    model.load()

    score_hybrid = model.predict(imgage_hybrid)
    score_non_hybrid = model.predict(image_non_hybrid)

    print()
    print(f"Hybrid butterfly score: {score_hybrid}")
    print(f"Non-hybrid butterfly score: {score_non_hybrid}")


if __name__ == "__main__":
    main()
