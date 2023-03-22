import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from project.src.base import Base
from project.constants import PHARMACOGENETIC_DOSING_ALGORITHM_COLUMNS


class PharmacogeneticDosing (Base):
    def __init__(self, perform_logging: bool = True) -> None:
        super().__init__(perform_logging)

    def set_data(self, df: pd.DataFrame):
        self.data = df[PHARMACOGENETIC_DOSING_ALGORITHM_COLUMNS].copy()
        self.gts = df[['GT']]

    def train(self) -> None:
        for t in tqdm(self.gts.index, 'Training'):
            if self.perform_logging and t > 0:
                self.log(t)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        rs9923231_factor = (
            -0.8677 * (x['rs9923231'] == 'A/G')
            + -1.6974 * (x['rs9923231'] == 'A/A')
        )
        rs9923231_factor = rs9923231_factor.fillna(-0.4854)

        Cyp2C9_factors = (
            -0.5211 * (x['Cyp2C9 genotypes'] == '*1/*2')
            + -0.9357 * (x['Cyp2C9 genotypes'] == '*1/*3')
            + -1.0616 * (x['Cyp2C9 genotypes'] == '*2/*2')
            + -1.9206 * (x['Cyp2C9 genotypes'] == '*2/*3')
            + -2.3312 * (x['Cyp2C9 genotypes'] == '*3/*3')
        )
        Cyp2C9_factors = Cyp2C9_factors.fillna(-0.2188)

        weekly_dose = (
            5.6044
            + -0.2614 * x['Age bucket']
            + 0.0087 * x['Height (cm)']
            + 0.0128 * x['Weight (kg)']
            + -0.1092 * x['Race_Asian']
            + -0.2760 * x['Race_Black or African American']
            + -0.1032 * x['Race_Unknown']
            + 1.1816 * x['Enzyme inducer status']
            + -0.5503 * x['Amiodarone (Cordarone)']
            + rs9923231_factor
            + Cyp2C9_factors
        ) ** 2

        preds = (
            0 * (weekly_dose < 21)
            + 1 * (weekly_dose <= 49)
            + 2 * (weekly_dose > 49)
        )

        return preds
