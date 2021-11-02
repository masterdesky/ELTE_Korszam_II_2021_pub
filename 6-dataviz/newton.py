import os
import numpy as np

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

#######
#
#    NEWTON-RAPHSON METHOD IMPLEMENTATION
#
##########################################################################

def NR_step(P, x):
  return x - P.f(x)/P.fprime(x)
def NR_iter(P, x, N):
  Y = x.copy()
  for i in range(N):
    Y = NR_step(P, Y)
  return Y



#######
#
#    FUNCTIONS TO ANALYZE WITH THE NEWTON-RAPHSON METHOD
#
##########################################################################

# Demo function with real roots
def demo_f(x):
  return 2*x**3 + 5*x**2 - 14*x - 2
def demo_fprime(x):
  return 6*x**2 + 10*x - 14

# Demo function with real and complex roots
def demo_fz(z):
  return z**5 + z**2 - z + 1
def demo_fzprime(z):
  return 5*z**4 + 2*z - 1



#######
#
#    NEWTON-RAPHSON INTRO
#
##########################################################################

def get_tangent_line(P, x):
  
  m = P.fprime(x)
  b = P.f(x) - m*x
  
  return m, b



#######
#
#    NEWTON-FRACTAL INTRODUCTION
#
##########################################################################

def get_starting_grid(N, grid_lim_x, grid_lim_y):

  X = np.meshgrid(np.linspace(*grid_lim_x, N), np.linspace(*grid_lim_y, N))
  X = X[0].flatten() + X[1].flatten()*1j
  
  return X



#######
#
#    NEWTON-FRACTAL COLORING
#
##########################################################################
  
def closest_roots(X, roots):
  
  a = np.array([roots - x for x in X])
  closest = np.array(
    [np.argmin(np.abs(i)) for i in a]
  )
  return closest

def get_cmap():  
  return sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
  #return sns.color_palette("ch:start=-0.2,rot=-0.1", as_cmap=True)
  #return cm.bone
  
def get_NR_colors(P, X):
  
  # Get roots of the polynomial
  roots = P.roots()
  # Find closest root to each grid points
  closest = closest_roots(X, roots)

  # Get colors according to closest root
  cmap = get_cmap()
  colors = cmap(closest_roots(X, roots)/(len(P.coeff_())-1))

  return colors



#######
#
#    OTHER FUNCTIONS
#
##########################################################################

def NR_missing_grid_lim(P):
  
  # Get roots of the polynomial
  roots = P.roots()
  
  lim = np.max((np.abs(roots.real), np.abs(roots.imag))) * 1.15
  
  return tuple((-lim, lim))

def NR_fractal_get_frames(gl_s, gl_e, n=50):
  
  v_s = np.array(list(product(*gl_s)))
  v_e = np.array(list(product(*gl_e)))

  v_coords = np.linspace(v_s, v_e, n)
  
  return v_coords

def NR_fractal_get_grid_lims(v_coords):
  
  # Shorten variable names
  v1, v2 = v_coords[:,0], v_coords[:,3]
  
  grid_lims =\
  tuple(
    tuple(
      ((xmin, xmax),(ymin, ymax))
    )
    for xmin, xmax, ymin, ymax in zip(v1[:,0], v2[:,0], v1[:,1], v2[:,1])
  )
  
  return grid_lims

def is_square(v):
  """
  v : array-like of shape (4,2)
    Coordinates of the vertices of a 2D rectangle
  """
  d = np.linalg.norm
  c = np.isclose
  rtol = 1e-07
  atol = 1e-08

  # Side lengths of the rectange
  ds = (d(v[0]-v[1]), d(v[1]-v[3]),
        d(v[3]-v[2]), d(v[2]-v[0]))

  if np.all([c(ds[0], o, rtol, atol) for o in ds[1:]]):
    return True
  return False

def get_data(n_samples=1000, noise=20):

  X, y, coef = make_regression(n_samples=n_samples, n_features=1,
                               n_informative=1, noise=noise,
                               coef=True, random_state=0)
  X = X.reshape((-1,1))
  y = y.reshape((-1,1))
  X_s = MinMaxScaler().fit_transform(X)
  y_s = MinMaxScaler().fit_transform(y)

  return np.c_[X_s,y_s]