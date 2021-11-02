import os
import sys
import jdcal
import numpy as np
import pandas as pd
from scipy import interpolate

from colorcet import palette

import datashader as ds
import datashader.colors as dc
import datashader.utils as utils
from datashader import transfer_functions as tf

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.dates as mdates
# FUTURE WARNING: EXPLICIT CONVERTER DECLARATION REQUIRED
# WHEN PLOTTING 'DATETIMES' FROM PANDAS DATAFRAMES
# USING MATPLOTLIB
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# Format displayed ticks as years
dates_fmt = mdates.DateFormatter('%Y-%m-%d')
# Put at least `minticks` dates on the X axis when plotting
dates_loc = mdates.AutoDateLocator(minticks=8)


#######
#
#    INTRODUCTION PLOTS
#
##########################################################################
def generate_random(S=1.0):
  """
  S : float, default: 1.0
    Scale parameter
  """
  sign = np.random.choice([-1,1])

  return np.random.random() * S * sign


def plot_curve(ax):
  
  signal = np.sin(np.linspace(0,8*np.pi,1000))
  slope = -0.17 * np.linspace(0,8*np.pi,1000) + 3
  noise = np.random.normal(loc=1.0, scale=1.8, size=1000)
  
  ax.plot(slope + signal + noise,
          color='tab:red')
  
  ax.set_ylim(-5,9)
  
  ax.set_title('a) Valamilyen jelfeldolgozás\nbeadandó egyik feladata',
               fontsize=18,
               y=-0.2)
  
  return ax

def plot_scatter(ax):

  # Generate sample data
  n_samples = 4000
  n_blobs = 4
  #centers = [[generate_random(S=7) for i in range(2)] for i in range(n_blobs)]
  centers = np.array((
    [5.1, -7.2],
    [-6.3, -2.2],
    [-3.2, 6.3],
    [3.2, 0.4],
  ))
  
  X, y_true = make_blobs(n_samples=n_samples,
                         centers=centers,
                         cluster_std=2.0,
                         center_box=(-10.0, 10.0),
                         random_state=0)
  X = X[:, ::-1]
  
  colors = ['#4EACC5', '#FF9C34', '#4E9A06', 'm']

  for k, col in enumerate(colors):
    cluster_data = y_true == k
    ax.scatter(X[cluster_data, 0], X[cluster_data, 1],
               c=col, marker='.', s=10)
      
  ax.set_xlim(-12,12)
  ax.set_ylim(-12,12)
  
  ax.set_title('b) Klaszterezhető adatok',
               fontsize=18,
               y=-0.2)
      
  return ax

def complex_plot(ax):
  """
  Ripped from:
  https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html#sphx-glr-auto-examples-calibration-plot-calibration-multiclass-py
  """
  X, y = make_blobs(n_samples=2000, n_features=2, centers=3, random_state=42,
                  cluster_std=5.0)
  X_train, y_train = X[:600], y[:600]
  X_valid, y_valid = X[600:1000], y[600:1000]
  X_train_valid, y_train_valid = X[:1000], y[:1000]
  X_test, y_test = X[1000:], y[1000:]

  
  clf = RandomForestClassifier(n_estimators=25)
  clf.fit(X_train, y_train)
  cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
  cal_clf.fit(X_valid, y_valid)
  
  colors = ["r", "g", "b"]

  clf_probs = clf.predict_proba(X_test)
  cal_clf_probs = cal_clf.predict_proba(X_test)
  # Plot arrows
  for i in range(clf_probs.shape[0]):
      ax.arrow(clf_probs[i, 0], clf_probs[i, 1],
               cal_clf_probs[i, 0] - clf_probs[i, 0],
               cal_clf_probs[i, 1] - clf_probs[i, 1],
               color=colors[y_test[i]], head_width=1e-2)

  # Plot perfect predictions, at each vertex
  ax.plot([1.0], [0.0], 'ro', ms=20, label="Class 1")
  ax.plot([0.0], [1.0], 'go', ms=20, label="Class 2")
  ax.plot([0.0], [0.0], 'bo', ms=20, label="Class 3")

  # Plot boundaries of unit simplex
  ax.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

  # Annotate points 6 points around the simplex, and mid point inside simplex
  ax.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
               xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
               arrowprops=dict(facecolor='black', shrink=0.05),
               horizontalalignment='center', verticalalignment='center')
  ax.plot([1.0/3], [1.0/3], 'ko', ms=5)
  ax.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
              xy=(.5, .0), xytext=(.5, .1), xycoords='data',
              arrowprops=dict(facecolor='black', shrink=0.05),
              horizontalalignment='center', verticalalignment='center')
  ax.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
              xy=(.0, .5), xytext=(.1, .5), xycoords='data',
              arrowprops=dict(facecolor='black', shrink=0.05),
              horizontalalignment='center', verticalalignment='center')
  ax.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
              xy=(.5, .5), xytext=(.6, .6), xycoords='data',
              arrowprops=dict(facecolor='black', shrink=0.05),
              horizontalalignment='center', verticalalignment='center')
  ax.annotate(r'($0$, $0$, $1$)',
              xy=(0, 0), xytext=(.1, .1), xycoords='data',
              arrowprops=dict(facecolor='black', shrink=0.05),
              horizontalalignment='center', verticalalignment='center')
  ax.annotate(r'($1$, $0$, $0$)',
              xy=(1, 0), xytext=(1, .1), xycoords='data',
              arrowprops=dict(facecolor='black', shrink=0.05),
              horizontalalignment='center', verticalalignment='center')
  ax.annotate(r'($0$, $1$, $0$)',
              xy=(0, 1), xytext=(.1, 1), xycoords='data',
              arrowprops=dict(facecolor='black', shrink=0.05),
              horizontalalignment='center', verticalalignment='center')
  # Add grid
  ax.grid(False)
  for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ax.plot([0, x], [x, 0], 'k', alpha=0.2)
    ax.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
    ax.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

  ax.set_xlabel('Probability class 1', fontsize=15, fontweight='bold')
  ax.set_ylabel('Probability class 2', fontsize=15, fontweight='bold')
  ax.set_xlim(-0.05, 1.05)
  ax.set_ylim(-0.05, 1.05)
  _ = ax.legend(loc="best")
  
  ax.set_title('c) Nagyon bonyolult dolog fú de',
               fontsize=18,
               y=-0.2)
  
  return ax

def intro_figures():
  nrows = 1
  ncols = 3
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*9, nrows*7),
                           facecolor='0.95')
  fig.subplots_adjust(wspace=0.2)

  functions = (plot_curve, plot_scatter, complex_plot)
  for i, ax in enumerate(axes.reshape(-1)):
    ax.set_aspect('auto')
    functions[i](ax)

  plt.suptitle('Fig. 1. Különböző típusú adatvizualiációk amik pl. fizika ' +
               'szakon előkerülhetnek.',
               fontsize=20, fontweight='bold', y=-0.1)

  plt.show()


#######
#
#    MOTIVATION PLOTS
#
##########################################################################
  
def open_betelgeuse_data(data_file):

  with open(data_file, 'r') as f:
    star_data = pd.read_csv(f)

  return star_data

def format_dates(star_data):
  # Convert Julian Calendar Dates to Gregorian dates
  MJD_0 = 2400000.5
  dates = star_data[(star_data['Band'] == 'V')]['JD'].values - MJD_0

  greg_dates = []
  for date in dates:
    date = jdcal.jd2gcal(MJD_0, date)

    time = date[3] * 24 * 60
    hrs, sec = divmod(time * 60, 3600)
    mins, sec = divmod(sec, 60)
    sec = int(sec)

    date = '{0:02}-{1:02}-{2:02} {3:02.0f}:{4:02.0f}:{5:02.0f}'.format(date[0], date[1], date[2],
                                                                       hrs, mins, sec)
    greg_dates.append(pd.Timestamp(date))
    
  return greg_dates

def create_date_ticks(dates, N=22):
    """
    Create a list of string date ticks to use them on a figure.
    Empirical observation, that here 22 X-ticks should be on the
    figures for them to look nicely.
    """
    date_ticks = pd.date_range(min(dates), max(dates), periods=N).to_numpy()
    
    return date_ticks

def set_colors(arr, cmap, vmin=0, vmax=1):
    
  assert vmin >= 0, "Parameter \'vmin\' should be >=0 !"
  assert vmax <= 1, "Parameter \'vmax\' should be <=1 !"

  m = interpolate.interp1d([np.min(arr), np.max(arr)], [vmin, vmax])
  colors = cmap(m(arr))

  return colors

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    #items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def crappy_plot(star_data, ax):  
  v_mag = np.array(
    star_data[(star_data['Band'] == 'V')]['Magnitude'].values,
    dtype='float64'
  )
  ax.plot(v_mag)
  
  ax.set_xlabel('Time', fontsize=20)
  ax.set_ylabel('Magnitude', fontsize=20)
  
  return ax

def cool_plot(star_data, greg_dates, ax):
  
  # Set ax element sizes
  axtitlesize = 22
  axticksize = 17
  axlabelsize = 26
  axlegendsize = 23

  sradius = 7

  # Magnitude
  ax.invert_yaxis()
  ax.grid(True, color='0.7', ls='--', lw=1, alpha=0.38)

  v_mag = np.array(
    star_data[(star_data['Band'] == 'V')]['Magnitude'].values,
    dtype='float64'
  )
  v_mag_err = np.array(
    star_data[(star_data['Band'] == 'V')]['Uncertainty'].values,
    dtype='float64'
  )

  colors = set_colors(v_mag, cmap=cm.magma_r, vmin=0, vmax=0.5)
  
  ax.scatter(greg_dates, v_mag,
             color=colors, s=sradius**2)
  ax.errorbar(greg_dates, v_mag, yerr=v_mag_err,
              linestyle='None',
              ecolor=colors, alpha=0.6)

  # These can be set manually
  ax.set_ylim(1.81, 0.59)
  
  # Source text
  ax.text(x=0.1, y=-0.2, s='Source of data: https://www.aavso.org/',
          c='white', fontsize=15, fontweight='book',
          horizontalalignment='center', verticalalignment='center',
          transform=ax.transAxes,
          bbox=dict(facecolor='black', alpha=0.2, lw=0))

  ax.set_title('AAVSO light curve of Betelgeuse',
               fontsize=axtitlesize, fontweight='bold', color='white')

  #ax.set_xlabel('Time',
  #              fontsize=axlabelsize, color='white')
  ax.set_ylabel('V-band magnitude [mag]',
                fontsize=axlabelsize, color='white')
  
  # Set axis ticks
  major_size = 6
  major_width = 1.2
  ax.tick_params(axis='x', which='major', direction='in',
                 labelsize=axticksize-3, pad=10, colors='white',
                 length=major_size, width=major_width)
  ax.tick_params(axis='y', which='major', direction='in',
                 labelsize=axticksize, pad=10, colors='white',
                 length=major_size, width=major_width)
  
  # Set axis spine colors
  for k in ax.spines.keys(): ax.spines[k].set_color('white')
  
  ax.tick_params(axis='x', which='minor', bottom=False, colors='white')
  ax.tick_params(axis='y', which='minor', left=False, colors='white')
  

  # DATE FORMATTING SOURCE:
  #   - https://matplotlib.org/gallery/text_labels_and_annotations/date.html
  # Format the ticks by calling the locator instances of matplotlib.dates
  date_ticks = create_date_ticks(greg_dates, N=8)
  ax.set_xticks(date_ticks)
  ax.set_xticklabels(date_ticks, rotation=42, ha='center')
  ax.xaxis.set_major_formatter(dates_fmt)
  
  # Should be placed after setting x-ticks!!!
  ax.set_xlim(greg_dates[0], greg_dates[-1])

  return ax
  
def compare_plots(star_data, greg_dates):
  nrows = 1
  ncols = 2
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*15,nrows*10),
                           facecolor='white')
  fig.subplots_adjust(wspace=0.25)
  
  # Crappy plot
  ax = axes[0]
  ax = crappy_plot(star_data, ax)  

  # Cool plot
  ax = axes[1]
  ax.set_facecolor('black')
  ax = cool_plot(star_data, greg_dates, ax)

  extent = Bbox(full_extent(axes[1], 0.12))
  # It's best to transform this back into figure coordinates. Otherwise, it
  # won't behave correctly when the size of the plot is changed.
  extent = extent.transformed(fig.transFigure.inverted())
  # We can now make the rectangle in figure coords using the "transform" kwarg.
  rect = Rectangle([extent.xmin, extent.ymin], extent.width, extent.height,
                   facecolor='black', edgecolor='none', zorder=-1, 
                   transform=fig.transFigure)
  fig.patches.append(rect)

  plt.show()