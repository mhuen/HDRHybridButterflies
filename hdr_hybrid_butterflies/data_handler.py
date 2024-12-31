import os
import pandas as pd
from PIL import Image


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
