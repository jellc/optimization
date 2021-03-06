import numpy as np
import matplotlib.pyplot as plt

import frame


class State(frame.State):
    def __init__(self, init):
        super().__init__(init)

    def iterate(self):
        rho = 0.5
        c_1 = 0.0001
        c_2 = 0.9
        cur = self.current
        eva = self.evaluate()
        grad = self.grad()
        step = -grad
        while 1:
            nxt = State(cur + step)
            if nxt.evaluate() - eva <= c_1 * np.inner(step, grad) \
                and np.inner(nxt.grad() - c_2 * grad, step) >= 0:
                break
            step *= rho
        self.current += step


def run(init: np.ndarray, output):
    frame.run(State, init, output)
