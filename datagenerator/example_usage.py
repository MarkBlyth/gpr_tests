import neuronmodels as neuron
import matplotlib.pyplot as plt
import numpy as np


def main():
    fig, ax = plt.subplots()
    ts, ys = neuron.simple_data_generator(neuron.hodgkin_huxley, rtol=1e-6, observation_noise=1)
    ax.plot(ts, ys)
    plt.show()


if __name__ == "__main__":
    main()
