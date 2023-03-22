import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from project.constants import ES_DOSING_ALGORITHM_COLUMNS
from project.src.base import Base
from project.utils import get_argmax, get_reward


class EnsembleSampling(Base):
    def __init__(self, n_arms: int, num_models: int, variance: float, perform_logging: bool = True) -> None:
        super().__init__(perform_logging)

        self.num_models = num_models
        self.n_arms = n_arms
        self.variance = variance

        self.n_features = len(ES_DOSING_ALGORITHM_COLUMNS)

        self.theta = np.zeros((num_models, n_arms, self.n_features, 1))  # Mx3x8x1

        self.mu = np.zeros((self.n_arms, self.n_features, 1))  # 3x8x1
        self.Sigma = np.stack([self.get_variance_identity(variance) for _ in range(n_arms)])  # 3x8x8

        self.W = np.zeros((self.n_features, 1))  # 8x1

    def set_data(self, df: pd.DataFrame) -> None:
        self.data = df[ES_DOSING_ALGORITHM_COLUMNS].copy()
        self.gts = df[['GT']]

    def get_variance_identity(self, variance) -> np.ndarray:
        return variance * np.identity(self.n_features)

    def train(self) -> None:
        np.random.seed(2023)

        # Sample
        for m in range(self.num_models):
            for a in range(self.n_arms):
                sample = np.random.multivariate_normal(self.mu[a].squeeze(), self.Sigma[a])
                self.theta[m][a]= np.expand_dims(sample, -1)

        for t in tqdm(self.gts.index, 'Training'):
            x = np.expand_dims(self.data.loc[t].to_numpy(), -1)
            gt = self.gts.loc[t, 'GT']

            self.sample_m = np.random.randint(1, self.num_models)

            # Act
            list_arms = (self.theta[self.sample_m].squeeze(-1) @ x).squeeze(-1)
            chosen_a = get_argmax(list_arms)

            # Observe (reward)
            r = get_reward(chosen_a, gt)

            # Update
            inv_sigma = np.linalg.inv(self.Sigma[chosen_a])  # Inverse of Sigma at timestep t

            self.Sigma[chosen_a] = np.linalg.inv(inv_sigma + (x @ x.T) / self.variance)  # Sigma at timestep t+1

            for m in range(self.num_models):
                self.W = np.random.multivariate_normal(self.mu[a].squeeze(), self.get_variance_identity(self.variance))
                self.W = np.expand_dims(self.W, -1)

                self.theta[m][chosen_a] = self.Sigma[chosen_a] @ (
                    inv_sigma @ self.theta[m][chosen_a] + x * (r + self.W) / self.variance
                )  # 8x1 = 8x8 X (8x8 X 8x1 + 8x1*(1 + 8x1 / 1)) -- broadcasting

            if self.perform_logging and t > 0:
                self.log(t)

    def predict(self, x: pd.Series) -> np.ndarray:
        x = x.to_numpy()  # Tx8
        self.sample_m = np.random.randint(1, self.num_models)
        list_arms = (x @ self.theta[self.sample_m].squeeze(-1).T)

        return get_argmax(list_arms, axis=1)
