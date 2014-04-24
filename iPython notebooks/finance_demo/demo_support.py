from numpy import nan
from numpy.random import randn
import numpy as np
np.set_printoptions(precision=4)


from pandas.core.common import adjoin
from pandas import *
from pandas.io.data import DataReader
import pandas.util.testing as tm
tm.N = 10

import scikits.statsmodels.api as sm
import scikits.statsmodels.datasets as datasets

from qqplot import *

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt

# import pandas.rpy.common as com

# baseball = com.load_data('baseball', package='plyr')
# ff = com.load_data('french_fries', package='reshape2')

plt.rc('figure', figsize=(10, 6))
np.random.seed(123456)

panel = Panel.load('data_panel')
# panel = panel.drop(['IBM'], axis='minor')

close_px = panel['Adj Close']

# convert closing prices to returns
rets = close_px / close_px.shift(1) - 1

index = (1 + rets).cumprod()

eom_index = index.asfreq('EOM')

eom_index = index.asfreq('EOM', method='ffill')

def side_by_side(*objs, **kwds):
    space = kwds.get('space', 4)

    reprs = [repr(obj).split('\n') for obj in objs]
    print adjoin(space, *reprs)


def plot_corr(rrcorr, title=None, normcolor=False):
    #rrcorr[range(nvars), range(nvars)] = np.nan
    xnames = ynames = list(rrcorr.index)
    nvars = len(xnames)

    if title is None:
        title = 'Correlation Matrix'
    if normcolor:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = None, None

    fig = plt.figure()
    ax = fig.add_subplot(111)
    axim = ax.imshow(rrcorr.values, cmap=plt.cm.jet,
                     interpolation='nearest',
                     extent=(0,nvars,0,nvars), vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(nvars)+0.5)
    ax.set_yticklabels(ynames[::-1], fontsize='small',
                       horizontalalignment='right')
    ax.set_xticks(np.arange(nvars)+0.5)
    ax.set_xticklabels(xnames, fontsize='small',rotation=45,
                       horizontalalignment='right')
    #some keywords don't work in previous line ?
    plt.setp( ax.get_xticklabels(), fontsize='small', rotation=45,
             horizontalalignment='right')
    fig.colorbar(axim)
    ax.set_title(title)

def plot_acf_multiple(ys, lags=20):
    """

    """
    from scikits.statsmodels.tsa.stattools import acf
    # hack
    old_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 8

    plt.figure(figsize=(10, 10))
    xs = np.arange(lags + 1)

    acorr = np.apply_along_axis(lambda x: acf(x, nlags=lags), 0, ys)

    k = acorr.shape[1]
    for i in range(k):
        ax = plt.subplot(k, 1, i + 1)
        ax.vlines(xs, [0], acorr[:, i])

        ax.axhline(0, color='k')
        ax.set_ylim([-1, 1])

        # hack?
        ax.set_xlim([-1, xs[-1] + 1])

    mpl.rcParams['font.size'] = old_size

def load_mplrc():
    import re

    path = 'matplotlibrc'
    regex = re.compile('(.*)[\s]+:[\s]+(.*)[\s]+#')
    for line in open(path):
        m = regex.match(line)
        if not m:
            continue

        cat, attr = m.group(1).strip().rsplit('.', 1)
        plt.rc(cat, **{attr : m.group(2).strip()})


# load_mplrc()
