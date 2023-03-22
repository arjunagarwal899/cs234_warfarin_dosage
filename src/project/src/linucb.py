import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from project.constants import LINUCB_DOSING_ALGORITHM_COLUMNS
from project.src.base import Base
from project.utils import get_argmax, get_reward


class LinUCB (Base):
    def __init__(self, n_arms: int, alpha: float, perform_logging: bool = True) -> None:
        super().__init__(perform_logging)

        self.alpha = alpha
        self.n_arms = n_arms

        n_features = len(LINUCB_DOSING_ALGORITHM_COLUMNS)

        self.A = np.stack([np.identity(n_features) for _ in range(n_arms)])  # 3x8x8
        self.b = np.zeros((n_arms, n_features, 1))  # 3x8x1
        self.theta = np.zeros((n_arms, n_features, 1))  # 3x8x1
        self.p = np.zeros((n_arms,))  # 3x0

    def set_data(self, df: pd.DataFrame) -> None:
        self.data = df[LINUCB_DOSING_ALGORITHM_COLUMNS].copy()
        self.gts = df[['GT']]

    def train(self) -> None:
        np.random.seed(2023)

        for t in tqdm(self.gts.index, 'Training'):
            x = np.expand_dims(self.data.loc[t].to_numpy(), -1)
            gt = self.gts.loc[t, 'GT']

            for a in range(self.n_arms):
                A_inv = np.linalg.inv(self.A[a])
                self.theta[a] = A_inv @ self.b[a]
                self.p[a] = self.theta[a].T @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)

            chosen_a = get_argmax(self.p)
            r = get_reward(chosen_a, gt)

            self.A[chosen_a] += x @ x.T
            self.b[chosen_a] += r * x

            if self.perform_logging and t > 0:
                self.log(t)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x = x.to_numpy()  # Tx8
        A_inv = np.linalg.inv(self.A)  # 3x8x8
        self.theta = A_inv @ self.b  # 3x8x1
        expected_r = x @ self.theta.squeeze().T  # Tx3
        return get_argmax(expected_r, axis=1)
