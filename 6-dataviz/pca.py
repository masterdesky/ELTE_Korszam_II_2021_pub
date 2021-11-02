import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from newton import get_data


def get_pca(X):
  
  # Calculate principal components
  pca = PCA(n_components=2)
  pca.fit(X)
  
  return pca

def pca_ax1(ax, X):
  
  ax.scatter(X[:,0], X[:,1],
             c='black', ec='none', s=4**2, alpha=0.6, zorder=2)
  
  ax.tick_params(axis='both', which='both', labelcolor='black')
  
  ax.set_xlabel('X',
                fontsize=20, fontweight='bold')
  ax.set_ylabel('Y',
                fontsize=20, fontweight='bold')
  
  ax.set_xlim(-0.05,1.05)
  ax.set_ylim(-0.05,1.05)
  
  return ax

def pca_ax2(ax, X, pca):
  
  # Plot the data
  ax = pca_ax1(ax=ax, X=X)

  for l, pc in zip(pca.explained_variance_, pca.components_):
    v = pc * 3 * np.sqrt(l)

    # Plot principal components
    ax.arrow(*pca.mean_, *v,
             color='tab:red', lw=6, alpha=0.8, zorder=3)
 
  ax.set_xlabel('X',
                fontsize=20, fontweight='bold')
  ax.set_ylabel('Y',
                fontsize=20, fontweight='bold')

  ax.set_xlim(-0.05,1.05)
  ax.set_ylim(-0.05,1.05)
  
  return ax

def pca_ax3(ax, X, pca):
  
  X_pca = pca.transform(X)
  
  ax = pca_ax1(ax=ax, X=X_pca)
  
  # Get largest element
  lim = np.max(np.abs(X_pca))

  # Plot principal components
  ax.arrow(*[0,0], *[0.6*lim,0],
           color='tab:red', lw=6, alpha=0.8, zorder=3)
  ax.arrow(*[0,0], *[0,0.6*lim],
           color='tab:red', lw=6, alpha=0.8, zorder=3)
  
  ax.set_xlabel('1. főkomponens',
                fontsize=20, fontweight='bold')
  ax.set_ylabel('2. főkomponens',
                fontsize=20, fontweight='bold')
  
  pad = 0.05*lim
  ax.set_xlim(-lim-pad,lim+pad)
  ax.set_ylim(-lim-pad,lim+pad)
  
  return ax

def plot_pca(n_samples=1000, noise=20,
             pad=0.05):
  
  nrows, ncols = 1, 3
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8, nrows*8),
                           facecolor='0.95')
  fig.subplots_adjust(wspace=0.2)
  axes = axes.flatten()
  
  for ax in axes:
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major',
                   labelsize=16)
  
  X = get_data(n_samples=n_samples, noise=noise)
  pca = get_pca(X)
  _ = pca_ax1(ax=axes[0], X=X)
  _ = pca_ax2(ax=axes[1], X=X, pca=pca)
  _ = pca_ax3(ax=axes[2], X=X, pca=pca)
  
  plt.show()