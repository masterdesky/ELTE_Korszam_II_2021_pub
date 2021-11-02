import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from newton import get_data


def fig_parts_ax1(ax, X):

  ax.set_title('1. Színek',
               color='white', fontsize=18, fontweight='bold')
  
  ax.set_facecolor('lightgray')
  
  colors = cm.magma(X[:,0]*0.55+0.2)
  ax.scatter(X[:,0], X[:,1],
             c=colors, ec='none', s=7**2, alpha=0.6, zorder=2)
  ax.plot([0,1], [0,1],
          color='black', lw=6, alpha=0.6, zorder=3)

  ax.tick_params(axis='both', which='both', labelcolor='black')

  return ax

def fig_parts_ax2(ax):

  ax.set_title('2. Tengelyfeliratok',
               color='white', fontsize=18, fontweight='bold')
  ax.set_xlabel('Nagyon hosszú név, amit\nmuszáj két sorba törni',
                color='white', fontsize=18, fontweight='bold', labelpad=10)
  ax.set_ylabel('Valamilyen mennyiség $\\left[ \dfrac{kg \cdot m}{s^{2}} \\right]$',
                color='white', fontsize=18, fontweight='bold', labelpad=2)
  
  ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
  ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
  ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
  ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
  ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
  ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
  
  ax.tick_params(axis='both', which='major', labelcolor='white',
                 labelsize=16, pad=8)
  ax.tick_params(axis='both', which='minor', labelcolor='white',
                 labelsize=10, pad=5)
  ax.tick_params(axis='x', which='both', labelcolor='white',
                 rotation=50)
  
  return ax

def fig_parts_ax3(ax):
  
  ax.set_title('3. Rácsvonalak',
               color='white', fontsize=18, fontweight='bold')
  ax.xaxis.set_minor_locator(AutoMinorLocator(n=3))
  ax.yaxis.set_minor_locator(AutoMinorLocator(n=3))
  
  ax.grid(True, which='major', ls='--', lw=2, alpha=0.6)
  ax.grid(True, which='minor', ls='-.', lw=1, alpha=0.6)
  
  ax.tick_params(axis='both', which='both', labelcolor='black')
  
  return ax

def fig_parts_ax4(ax):
  
  ax.set_title('$\Leftarrow\Leftarrow\Leftarrow$ 4. Ábrák elrendezése!',
               color='white', fontsize=18, fontweight='bold')
  ax.tick_params(axis='both', which='both', colors='black')
  
  return ax

def fig_parts(n_samples=1000, noise=20):
  
  nrows, ncols = 1, 4
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8, nrows*6),
                           facecolor='black')
  fig.subplots_adjust(wspace=0.15)
  axes = axes.flatten()
  
  for ax in axes:
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', colors='white')
  
  X = get_data(n_samples=n_samples, noise=noise)
  _ = fig_parts_ax1(axes[0], X=X)
  _ = fig_parts_ax2(axes[1])
  _ = fig_parts_ax3(axes[2])
  _ = fig_parts_ax4(axes[3])
  
  plt.show()