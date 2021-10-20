from sampling.sampling import SamplingAlgorithm
from imblearn.over_sampling import ADASYN
import numpy as np
from typing import Tuple
import os
import shutil
import tempfile
import pandas as pd


class RobRoseAlgorithm(SamplingAlgorithm):

    @staticmethod
    def run(x_train: np.array, y_train: np.array) -> Tuple[np.array, np.array]:
        label_name = 'Label'
        r = 0.2
        alpha = 0.5
        const = 1
        seed = 4221
        x = pd.DataFrame(x_train)
        y = pd.DataFrame(y_train, columns=[label_name])
        df = x.join(y)
        tmp_input = tempfile.NamedTemporaryFile(suffix='.csv')
        os.unlink(tmp_input.name)
        df.to_csv(tmp_input.name, index=False)

        tmp_output = tempfile.NamedTemporaryFile(suffix='.csv')
        os.system(f'Rscript --vanilla sampling/sampling_robrose.R --file={tmp_input.name} --out={tmp_output.name} --label={label_name} --r={r} --alpha={alpha} --const={const} --seed={seed}')
        out_df = pd.read_csv(out, index_col=0)

    

        # balanced_x, balanced_y = ADASYN().fit_resample(x_train, y_train)
        # return balanced_x, balanced_y
