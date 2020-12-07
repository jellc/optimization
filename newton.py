import numpy as np
import matplotlib.pyplot as plt

import frame


class State(frame.State):
    def __init__(self, init):
        super().__init__(init)

    def iterate(self):
        self.current -= np.array(np.matrix(self.grad()) * np.linalg.inv(self.hessian()))
        self.current = list(self.current[0])


def run(init: np.ndarray, output):
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
