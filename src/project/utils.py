import numpy as np

from scipy.stats import t
from matplotlib import pyplot as plt


def get_performances(predictions: list, targets: list) -> dict:
    assert len(predictions) == len(targets), 'Length of predictions and targets does not match'

    performances = {}

    frac_incorrect = ((np.array(predictions) != np.array(targets)) * 1).sum() / len(predictions)
    performances['fraction of incorrect doses']= frac_incorrect

    regrets = np.array([
        get_reward(t, t) - get_reward(p, t) for p, t in zip(predictions, targets)
    ])
    performances['cumulative regret'] = regrets.sum()
    # performances['average regret'] = regrets.mean()

    return performances

def get_argmax(arr: np.ndarray, axis: int = None) -> int:
    assert axis is not None or len(arr.shape) == 1, 'Cannot get argmax of arays having multiple dimensions'
    if axis is None:
        axis = 0
    return np.argmax(np.random.random(arr.shape) * (arr==np.amax(arr, axis=axis, keepdims=True)), axis=axis)

def get_reward(prediction: int, groundtruth: int) -> int:
    # if prediction - groundtruth == 2:
    #     return -50
    # if groundtruth - prediction == 2:
    #     return -20
    # elif abs(prediction - groundtruth) == 1:
    #     return -10
    # return 0

    if prediction == groundtruth:
        return 0
    return -1

def get_confidence(data: np.ndarray, alpha: float = 0.025):
    x_bar = np.mean(data, axis=0)
    s = np.std(data, axis=0, ddof=1)
    
    dof = len(data) - 1
    t_star = t.ppf(alpha, dof)
    
    CI = t_star * s / np.sqrt(len(data))
    CI_upper = x_bar - CI
    CI_lower = x_bar + CI

    return x_bar, CI_lower, CI_upper

def plot_performances(data: np.ndarray, label: str, xlabel: str, ylabel: str, title: str) -> None:
    if data.ndim == 2:
        # min_limit = data.min(axis=0)
        # max_limit = data.max(axis=0)
        # plt.fill_between(range(len(data[0])), min_limit, max_limit, alpha = 0.5)

        _, CI_lower, CI_upper = get_confidence(data)
        plt.fill_between(range(len(data[0])), CI_lower, CI_upper, alpha = 0.5)
        
        data = data.mean(axis=0)

    plt.plot(data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_ci(data: list, title: str, alpha: float = 0.025) -> None:
    colors = [
        'green',
        'red',
        'blue'
    ]

    for i, (color, (key, values)) in enumerate(zip(colors, data.items())):
        # Calculation of Confidence Intervall
        x_bar, CI_lower, CI_upper = get_confidence(values, alpha)

        # Plot configuration
        l = i - 0.1
        r = i + 0.1

        plt.plot(i, x_bar, 'o', color='black')
        
        plt.plot([i, i], [CI_upper, CI_lower], color=color, label=key)
        plt.plot([l, r], [CI_lower, CI_lower], color=color)
        plt.plot([l, r], [CI_upper, CI_upper], color=color)
        
    # plt.suptitle('95% confidence intervals')
    plt.title(title)
    plt.legend()
    plt.xticks([])
    plt.show()