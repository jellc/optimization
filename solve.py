import matplotlib.pyplot as plt
import numpy as np

import steepest
import newton
import quasi_newton


def main():
    steepest.run([1.2, 1.2], "out/steepest1.png")
    steepest.run([-1.2, 1.0], "out/steepest2.png")
    newton.run([1.2, 1.2], "out/newton1.png")
    newton.run([-1.2, 1.0], "out/newton2.png")
    # quasi_newton.run([1.2, 1.2], "quasi_newton1.png")
    # quasi_newton.run([-1.2, 1.0], "quasi_newton2.png")


if __name__ == "__main__":
    main()
