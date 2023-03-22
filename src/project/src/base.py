import numpy as np
import pandas as pd

from collections import defaultdict
from project.utils import get_performances, plot_performances


class Base:
    def __init__(self, perform_logging: bool = True) -> None:
        self.data = None
        self.gts = None
        self.performance_history = defaultdict(list)
        self.perform_logging = perform_logging

    def set_data(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def log(self, T: int) -> None:
        gts = self.gts.loc[:T].to_numpy().squeeze()
        preds = self.predict(self.data.loc[:T])

        performances = get_performances(preds, gts)
        for key, value in performances.items():
            self.performance_history[key].append(value)

    def plot_performances(self) -> None:
        class_name = self.__class__.__name__

        for key, history in self.performance_history.items():
            history = np.array(history)
            if key == 'fraction of incorrect doses':
                plot_performances(history, class_name, 'Time', 'Fraction', 'Fraction of incorrect doses')
            elif key == 'cumulative regret':
                plot_performances(history, class_name, 'Time', 'Regret', 'Cumulative regret')
            elif key == 'average regret':
                plot_performances(history, class_name, 'Time', 'Regret', 'Average regret')
            else:
                raise NotImplementedError

    def get_final_performances(self) -> dict:
        final_performances = {}
        for key, values in self.performance_history.items():
            final_performances[key] = values[-1]
        return final_performances
