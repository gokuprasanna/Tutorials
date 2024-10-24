"""
Scratch file
"""

import numpy as np
import matplotlib.pyplot as plt

def test_fn():
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x)
    plt.plot(y)
    plt.show()

if __name__ == "__main__":
    test_fn()
