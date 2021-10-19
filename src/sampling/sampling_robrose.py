import os

import pandas as pd
from sampling.sampling import SamplingAlgorithm

class RobRoseSamplingAlgorithm(SamplingAlgorithm):

    @staticmethod
    def run(file, out, label, r=0.2, alpha=0.5, const=1, seed=42):
        """Runs RobROSE algorithm on given dataset

        Args:
            file (str): Path to dataset
            out (str): Path to save resampled dataset
            label (str): Column name of class label
            r (float, optional): Desired fraction of minority class in original data. Defaults to 0.2.
            alpha (float, optional): Numeric parameter used by the covMcd function for controlling the size of the subsets over which the determinant is minimized. Defaults to 0.5.
            const (int, optional): Tuning constant that changes the volume of the elipsoids. Defaults to 1.
            seed (int, optional): A single value, interpreted as an integer, recommended to specify seeds and keep trace of the generated sample. Defaults to 42.

        Returns:
            pd.DataFrame: Dataframe of synthetic samples
        """
        os.system(f'Rscript --vanilla sampling/sampling_robrose.R --file={file} --out={out} --label={label} --r={r} --alpha={alpha} --const={const} --seed={seed}')
        out_df = pd.read_csv(out, index_col=0)
        return out_df
        
