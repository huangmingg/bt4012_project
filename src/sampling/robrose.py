from typing import Tuple, List
import os
import tempfile

import numpy as np
import pandas as pd

from sampling.sampling import SamplingAlgorithm



class RobRoseAlgorithm(SamplingAlgorithm):

    @staticmethod
    def run(x_train: np.array, y_train: np.array, columns: List[str], **kwargs) -> Tuple[np.array, np.array]:
        """Runs robROSE algorithm to balance given dataset

        Args:
            x_train (np.array): Array containing sample features, where shape is (n_samples, n_features)
            y_train (np.array): Target vector relative to x_train

            **label (str): Name of target variable
            **columns (list): List of column names for features
            **r (float): Desired fraction of minority class
            **alpha (float): Numeric parameter used by the covMcd function for controlling the size of the subsets over which the determinant is minimized
            **const (float): Tuning constant that changes the volume of the elipsoids
            **seed (int): "A single value, interpreted as an integer, recommended to specify seeds and keep trace of the generated sample.

        Returns:
            Tuple[np.array, np.array]: Tuple containing sample features and target vector of balanced dataset
        """

        label = 'class'
        r = kwargs['r']
        alpha = kwargs['alpha']
        const = kwargs['const']
        seed = kwargs['seed']
        x = pd.DataFrame(x_train, columns=columns)       
        y = pd.DataFrame(y_train, columns=[label])
        df = x.join(y)
        tmp_input = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        df.to_csv(tmp_input.name, index=False)

        tmp_output = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        os.system(f'Rscript --vanilla sampling/sampling_robrose.R --file={tmp_input.name} --out={tmp_output.name} --label={label} --r={r} --alpha={alpha} --const={const} --seed={seed}')
        print(tmp_output.name)
        tmp_output.close()
        sampled_df = pd.read_csv(tmp_output.name, index_col=0)
        balanced_df = pd.concat([df, sampled_df], axis=0)
        balanced_x = balanced_df.drop(label, axis=1).to_numpy()
        balanced_y = balanced_df[label].to_numpy()

        tmp_input.close()
        os.unlink(tmp_input.name)
        os.unlink(tmp_output.name)

        return balanced_x, balanced_y


