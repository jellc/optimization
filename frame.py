import matplotlib.pyplot as plt
import numpy as np


class State:
    def __init__(self, init: np.ndarray):
        self.current = init
        self.dimension = len(init)

    def evaluate(self):
        cur = self.current
        ret = 0
        i = 0
        while i + 1 < self.dimension:
            ret += (1 - cur[i])**2 + 100 * (cur[i + 1] - cur[i]**2)**2
            i += 1
        return ret

    def grad(self):
        cur = self.current
        ret = np.zeros(self.dimension)
        i = 0
        while i + 1 < self.dimension:
            ret[i] += (cur[i] - 1) * 2 + 400 * cur[i] * (cur[i]**2 - cur[i + 1])
            ret[i + 1] += 200 * (cur[i + 1] - cur[i]**2)
            i += 1
        return ret

    def hessian(self):
        cur = self.current
        ret = np.zeros(shape=(self.dimension, self.dimension))
        i = 0
        while i + 1 < self.dimension:
            ret[i][i + 1] -= float(cur[i]) * 400
            ret[i + 1][i] -= 400 * cur[i]
            ret[i][i] = ret[i][i] + 2 + cur[i]**2 * 1200
            ret[i + 1][i + 1] += 200
            i += 1
        return ret


def run(State, init: np.ndarray, output):
    iteration = 200

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    stat = State(init)
    goal = np.array([1, 1])
    diff = []
    dist = []
    for _ in range(iteration):
        stat.iterate()
        eva = stat.evaluate()
        diff.append(eva)
        dist.append(np.linalg.norm(stat.current - goal))

    ax1.set_xlabel('number of iterations', fontsize=14)
    ax1.set_ylabel('$||x_k-x||$', fontsize=14)
    ax1.set_title("disntance to opt", fontsize=14)
    ax1.plot(dist)

    ax2.set_xlabel('number of iterations', fontsize=14)
    ax2.set_ylabel('$f(x)$', fontsize=14)
    ax2.plot(diff)

    fig.savefig(output)
    plt.close(fig)
