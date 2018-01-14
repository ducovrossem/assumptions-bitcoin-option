import numpy as np
from scipy.stats.kde import gaussian_kde
from numpy import linspace
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

# -------------- Plotting Functions ----------------------

def plot_saver(pandas_plot_output, title, path = ''):
    fig = pandas_plot_output.get_figure()
    fig.savefig(path + title.replace(' ', '_') +  '.png')

def plot_retdist(retshist,
                 retshist2 = None,
                 norm = False,
                 add_legend = None,
                 save = False,
                 savepath = 'logs/',
                 min_upperlimit = None,
                 max_lowerlimit = None,
                 title = "Distribution of Returns"):
    """
    Plots return distribution using gaussian_kde for array of returns retshist and optionally retshist2.

    :param retshist: nparray with daily returns
    :param retshist2: nparray with daily returns
    :param norm: bool on whether to normalize kde output
    :param add_legend: None implies no legend. List of legend labels for rethist (and rethist2).
    :param save: bool.
    :param savepath: str.
    :param min_upperlimit: float.
    :param max_lowerlimit: float.
    :param title: str.
    :return:
    """

    kde_ret = gaussian_kde(retshist)
    plot_second_dist = retshist2 is not None

    if not plot_second_dist:
        dist_space = linspace(min(retshist), max(retshist), 100)
    else:
        # simple way to set min/max of figure:
        kde_ret2 = gaussian_kde(retshist2)
        if min_upperlimit is not None:
            upperlimit = min(max(max(retshist), max(retshist)), min_upperlimit)
        else:
            upperlimit = max(max(retshist), max(retshist))
        if max_lowerlimit is not None:
            lowerlimit = max(min(min(retshist), min(retshist)), max_lowerlimit)
        else:
            lowerlimit = min(min(retshist), min(retshist))

        dist_space = linspace(lowerlimit, upperlimit, 100)

    plt.figure()

    if norm:
        plt.plot(dist_space, kde_ret(dist_space) / sum(kde_ret(dist_space)))
        if plot_second_dist:
            plt.plot(dist_space, kde_ret2(dist_space) / sum(kde_ret2(dist_space)))
    else:
        plt.plot(dist_space, kde_ret(dist_space))
        if plot_second_dist:
            plt.plot(dist_space, kde_ret2(dist_space))
    if add_legend is not None:
        plt.legend(labels=add_legend)

    plt.title(title)
    if save:
        plt.savefig(savepath + title.replace(' ', '_') + '.png')
