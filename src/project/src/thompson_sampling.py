import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from project.constants import TS_DOSING_ALGORITHM_COLUMNS
from project.src.base import Base
from project.utils import get_argmax, get_reward


class ThompsonSampling (Base):
    def __init__(self, n_arms: int, v: float, perform_logging: bool = True) -> None:
        super().__init__(perform_logging)

        self.n_arms = n_arms

        n_features = len(TS_DOSING_ALGORITHM_COLUMNS)

        self.v = v

        self.B = np.stack([np.identity(n_features) for _ in range(n_arms)])  # 3x8x8
        self.mu_hat = np.zeros((self.n_arms, n_features, 1))  # 3x8x1
        self.f = np.zeros((self.n_arms, n_features, 1))  # 3x8x1

    def set_data(self, df: pd.Series) -> None:
        self.data = df[TS_DOSING_ALGORITHM_COLUMNS].copy()
        self.gts = df[['GT']]

    def get_sample(self):
        samples = np.array([
            np.random.multivariate_normal(self.mu_hat[a].squeeze(), (self.v ** 2) * np.linalg.inv(self.B[a]))
            for a in range(self.n_arms)
        ])
        return samples

    def train(self) -> None:
        np.random.seed(2023)

        for t in tqdm(self.gts.index, 'Training'):
            x = np.expand_dims(self.data.loc[t].to_numpy(), -1)
            gt = self.gts.loc[t, 'GT']

            sample = self.get_sample()
            
            list_arms = sample @ x  # 3x8 X 8x1 -> 3x1
            chosen_a = get_argmax(list_arms.squeeze())
            r = get_reward(chosen_a, gt)

            self.B[chosen_a] += x @ x.T  # 8x1 X 1x8 -> 8x8
            self.f[chosen_a] += x * r  # 8x1
            self.mu_hat[chosen_a] = np.linalg.inv(self.B[chosen_a]) @ self.f[chosen_a]  # 8x8 X 8x1 -> 8x1

            if self.perform_logging and t > 0:
                self.log(t)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x = x.to_numpy()  # Tx8
        sample = self.get_sample()  # 3x8
        list_arms = (x @ sample.T).squeeze()
        return get_argmax(list_arms, axis=1)
