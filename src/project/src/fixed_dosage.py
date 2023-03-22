import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from project.src.base import Base


class FixedDosage (Base):
    def __init__(self, fixed_arm: float, perform_logging: bool = True) -> None:
        super().__init__(perform_logging)
        self.fixed_arm = fixed_arm

    def set_data(self, df: pd.DataFrame):
        self.data = df
        self.gts = df[['GT']]

    def train(self) -> None:
        for t in tqdm(self.gts.index, 'Training'):
            if self.perform_logging and t > 0:
                self.log(t)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return pd.Series([self.fixed_arm for _ in range(len(x))])
