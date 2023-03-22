import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from project.src.base import Base
from project.constants import CLINICAL_DOSING_ALGORITHM_COLUMNS


class ClinicalDosing (Base):
    def __init__(self, perform_logging: bool = True) -> None:
        super().__init__(perform_logging)

    def set_data(self, df: pd.DataFrame):
        self.data = df[CLINICAL_DOSING_ALGORITHM_COLUMNS].copy()
        self.gts = df[['GT']]

    def train(self) -> None:
        for t in tqdm(self.gts.index, 'Training'):
            if self.perform_logging and t > 0:
                self.log(t)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        weekly_dose = pd.Series(
            4.0376
            + -0.2546 * x['Age bucket']
            + 0.0118 * x['Height (cm)']
            + 0.0134 * x['Weight (kg)']
            + -0.6752 * x['Race_Asian']
            + 0.4060 * x['Race_Black or African American']
            + 0.0443 * x['Race_Unknown']
            + 1.2799 * x['Enzyme inducer status']
            + -0.5695 * x['Amiodarone (Cordarone)']
        ) ** 2

        preds = (
            0 * (weekly_dose < 21)
            + 1 * (weekly_dose <= 49)
            + 2 * (weekly_dose > 49)
        )

        return preds
