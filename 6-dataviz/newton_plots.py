import os
import numpy as np

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from newton import *

out = './out/'
figsave_fmt = 'png'
figsave_dpi = 200


#######
#
#    NEWTON-RAPHSON GENERAL-PURPOSE PLOTTING FUNCTIONS
#
##########################################################################

def NR_coord_axis(ax, alpha=0.5):
  
  ax.axhline(0, color='grey', ls='--', lw=3, alpha=alpha)
  ax.axvline(0, color='grey', ls='--', lw=3, alpha=alpha)
  
  return ax


def NR_intro_roots(ax, P, alpha=0.8):

  # Get roots
  roots = P.roots()
  
  # Plot roots on a real or complex plane
  ax.scatter(x=roots.real,
             y=roots.imag,
             label='Zérushely',
             color='tab:green', ec='black', s=15**2, alpha=alpha,
             zorder=3)
  return ax, roots

def NR_colored_roots(ax, P, alpha=0.8):
  
  # Get roots
  roots = P.roots()
  
  # Get different colors for different roots
  cmap = get_cmap()
  colors = cmap(np.linspace(0,1,len(P.coeff_())-1))
  
  # Plot roots on a real or complex plane
  ax.scatter(x=roots.real,
             y=roots.imag,
             label='Zérushely',
             c=colors, ec='black', s=15**2, alpha=alpha,
             zorder=3)
  return ax, roots

#######
#
#    NEWTON-RAPHSON INTRO PLOTS
#
##########################################################################

#
# 1. Plot elements
#
def NR_intro_f(ax, P, x, alpha=1.0):
  
  ax.plot(x, P.f(x), label='${}$'.format(P.f_str()),
          color='royalblue', lw=4, alpha=alpha)  
  return ax

def NR_intro_fprime(ax, P, x, alpha=0.7):
  
  ax.plot(x, P.fprime(x), label='${}$'.format(P.fprime_str()),
          color='tab:orange', lw=4, ls='-.', alpha=alpha)
  return ax

def NR_intro_random_x(ax, P, x, alpha=0.8):

  ax.scatter(x=x,
             y=0,
             label='Tetszőleges pont',
             color='tab:red', ec='black', s=12**2, alpha=alpha, zorder=3)
  ax.axvline(x, color='tab:red', ls=':', lw=3, alpha=alpha*0.8)
  
  return ax
  
def NR_intro_tangent(ax, P, x, x_tg, alpha=0.8):
  
  # Tangent line
  m, b = get_tangent_line(P, x)  
  x_tg = np.linspace(*x_tg, 20)
  ax.plot(x_tg, x_tg*m + b,
          label='Érintő egyenes',
          color='tab:red', lw=4, ls='-', alpha=alpha)
  
  return ax

def NR_intro_tangent_ic(ax, P, x, alpha=0.8):
  
  m, b = get_tangent_line(P, x)
  # Where tangent intercept X-axis: x = -b/m
  ax.axvline(-b/m, color='black', ls='--', lw=3, alpha=alpha)
  
  return ax

def NR_intro_new_x(ax, P, x, alpha=0.8):
  
  # Get tangent line to P polynomial at place `x`
  m, b = get_tangent_line(P, x)
  ax.scatter(x=-b/m,
             y=0,
             label='Új kezdőpont',
             color='gold', ec='black', s=12**2, alpha=alpha, zorder=3)
  
  return ax

def NR_intro_arrow(ax, P, x, alpha):

  # Get tangent line to P polynomial at place `x`
  m, b = get_tangent_line(P, x)
  x_s, x_e = x, -b/m 
  l = x_e - x_s
  arrow_s = x_s+l/8
  arrow_e = l-3*l/8
  
  ax.arrow(x=arrow_s, y=0,
           dx=arrow_e, dy=0,
           lw=8, head_width=1.0, head_length=0.06,
           length_includes_head=True)
  
  return ax


#
# 2. Construct individual axes
#
def NR_ax_1(ax, P, x):

  ax = NR_coord_axis(ax, alpha=0.5)
  
  ax = NR_intro_f(ax, P, x, alpha=1.0)
  ax = NR_intro_fprime(ax, P, x, alpha=0.7)
  ax, _ = NR_intro_roots(ax, P,
                         alpha=0.8)
  
  ax.set_xlim(x[0]-(x[-1]-x[0])/6, x[-1]+(x[-1]-x[0])/6)

  ax.set_title('a) Hol vannak a zéruspontjai\nennek a random polinomnak?',
               fontsize=20, fontweight='bold')
  
  ax.tick_params(axis='both', which='major', labelsize=15)

  ax.legend(loc='upper left', fontsize=16)
  
  return ax

def NR_ax_2(ax, P, x):

  ax = NR_coord_axis(ax, alpha=0.5)
  
  ax = NR_intro_f(ax, P, x, alpha=1.0)
  ax = NR_intro_fprime(ax, P, x, alpha=0.3)
  ax, _ = NR_intro_roots(ax, P, alpha=0.8)
  ax = NR_intro_random_x(ax, P,
                         x=2.8,
                         alpha=0.8)
  ax = NR_intro_tangent(ax, P,
                        x=2.8,
                        x_tg=(2,3.15),
                        alpha=0.8)  
  
  ax.set_xlim(x[0], x[-1]+(x[-1]-x[0])/6)
  
  ax.set_title('b) Tetszőleges pont kiválasztása\nés érintő húzása az eredeti függvényhez',
               fontsize=20, fontweight='bold')
  
  ax.set_ylabel('$\Downarrow$', fontsize=60)  
  ax.tick_params(axis='both', which='major', labelsize=15)

  ax.legend(loc='upper left', ncol=2, fontsize=16)
  
  return ax

def NR_ax_3(ax, P, x):

  ax = NR_coord_axis(ax, alpha=0.5)
  
  ax = NR_intro_f(ax, P, x, alpha=1.0)
  ax = NR_intro_fprime(ax, P, x, alpha=0.3)
  ax, _ = NR_intro_roots(ax, P, alpha=0.8)
  ax = NR_intro_random_x(ax, P,
                         x=2.8,
                         alpha=0.3)
  ax = NR_intro_tangent(ax, P,
                        x=2.8,
                        x_tg=(2,3.15),
                        alpha=0.8)
  ax = NR_intro_tangent_ic(ax, P, x=2.8, alpha=0.8)
  
  ax.set_xlim(x[0], x[-1]+(x[-1]-x[0])/6)
  
  ax.set_title('c) Hol metszi az érintő vonal az x-tengely?',
               fontsize=20, fontweight='bold')

  ax.tick_params(axis='both', which='major', labelsize=15)

  ax.legend(loc='upper left', ncol=2, fontsize=16)
  
  return ax

def NR_ax_4(ax, P, x):

  ax = NR_coord_axis(ax, alpha=0.5)
  
  ax = NR_intro_f(ax, P, x, alpha=1.0)
  ax = NR_intro_fprime(ax, P, x, alpha=0.3)
  ax, _ = NR_intro_roots(ax, P, alpha=0.8)
  ax = NR_intro_random_x(ax, P,
                         x=2.8,
                         alpha=0.3)
  ax = NR_intro_tangent(ax, P,
                        x=2.8,
                        x_tg=(2,3.15),
                        alpha=0.3)
  ax = NR_intro_tangent_ic(ax, P, x=2.8, alpha=0.8)
  ax = NR_intro_new_x(ax, P, x=2.8, alpha=0.8)
  ax = NR_intro_arrow(ax, P, x=2.8, alpha=0.9)
  
  ax.set_xlim(x[0], x[-1]+(x[-1]-x[0])/6)
  
  ax.set_title('d) Az iteráció következő lépésének az új\nkezdőpont a bemenete',
               fontsize=20, fontweight='bold')

  ax.set_ylabel('$\Downarrow$', fontsize=60)  
  ax.tick_params(axis='both', which='major', labelsize=15)

  ax.legend(loc='upper left', ncol=2, fontsize=16)
  
  return ax


#
# 3. Build intro plot
#
def NR_intro(P):
  
  nrows = 2
  ncols = 2
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*10),
                           facecolor='0.96')

  axes = axes.flatten()
  _ = NR_ax_1(axes[0], P, x=np.linspace(-5,3,100))
  _ = NR_ax_2(axes[1], P, x=np.linspace(0,3,100))
  _ = NR_ax_3(axes[2], P, x=np.linspace(0,3,100))
  _ = NR_ax_4(axes[3], P, x=np.linspace(0,3,100))

  fig.suptitle('Fig. 3. Newton--Raphson-módszer motivációja ' +
               'és működése valós függvényekre',
               fontsize=20, fontweight='bold',
               y=0.09)

  plt.show()



#######
#
#    NEWTON-RAPHSON ON THE COMPLEX PLANE
#
##########################################################################

def NR_complex_fig_setup(nrows, ncols, grid_lim, axis=True):

  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*10),
                           facecolor='0.96')
  fig.subplots_adjust(wspace=0.15, hspace=0.15)
  
  if type(axes) is np.ndarray:
    axes = axes.flatten()
  else:
    axes = np.array((axes))
    
  for i, ax in enumerate(axes):
    ax.set_xlim(grid_lim)
    ax.set_ylim(grid_lim)
    
    if i+ncols >= len(axes):
      ax.set_xlabel('Real [$\mathfrak{R}$]', fontsize=25, fontweight='bold')
    if i%ncols == 0:
      ax.set_ylabel('Imag [$\mathfrak{I}$]', fontsize=25, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=15)

    if axis:
      ax = NR_coord_axis(ax, alpha=0.5)    
    
  return fig, axes

def NR_complex_ax1(ax, P, x):

  ax = NR_coord_axis(ax, alpha=0.5)
  ax.plot(x, P.f(x), label='${}$'.format(P.f_str()),
          color='royalblue', lw=4)
  ax.plot(x, P.fprime(x), label='${}$'.format(P.fprime_str()),
          color='tab:orange', lw=4, ls='-.', alpha=0.7)
  
  ax.set_xlabel('X', fontsize=25, fontweight='bold')
  ax.set_ylabel('Y', fontsize=25, fontweight='bold')
  
  ax.tick_params(axis='both', which='major', labelsize=15)

  ax.legend(loc='upper center', fontsize=18)
  
  return ax

def NR_complex_ax2(ax, P, grid_lim):
  
  ax = NR_coord_axis(ax, alpha=0.5)

  ax, _ = NR_intro_roots(ax, P, alpha=0.8)
  
  ax.set_xlim(*grid_lim)
  ax.set_ylim(*grid_lim)
  
  ax.set_xlabel('Real [$\mathfrak{R}$]', fontsize=25, fontweight='bold')
  ax.set_ylabel('Imag [$\mathfrak{I}$]', fontsize=25, fontweight='bold')
  
  ax.tick_params(axis='both', which='major', labelsize=15)
  
  ax.legend(loc='upper left', fontsize=18)
  
  return ax

def NR_complex_intro(P,
                     grid_lim=None):
  
  if grid_lim is None:
    grid_lim = NR_missing_grid_lim(P)
  
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2*10, 1*10),
                           facecolor='0.96')
  fig.subplots_adjust(wspace=0.25)
  
  x = np.linspace(-2, 2, 100)
  _ = NR_complex_ax1(ax=axes[0], P=P, x=x)
  _ = NR_complex_ax2(ax=axes[1], P=P, grid_lim=grid_lim)
  
  plt.suptitle('Fig. 4. Egy függvény, aminek vegyesen valós és komplex gyökei is vannak',
               fontsize=20, fontweight='bold', y=0.00)
  
  plt.show()


def NR_complex_annotate(ax, text, xy, xytext, alpha=1.0):
  
  ax.annotate(text,
              xy=(xy.real, xy.imag), xycoords='data', xytext=xytext,
              fontsize=28, horizontalalignment='right', verticalalignment='top',
              alpha=alpha
             )
  
  return ax

def NR_complex_points(ax, P, x, idx=0, alpha=1.0):
  
  ax.scatter(x.real, x.imag,
             color='tab:red', s=12**2, alpha=alpha)
  ax.scatter(P.f(x).real, P.f(x).imag,
             color='mediumorchid', s=12**2, alpha=alpha)
  
  ax = NR_complex_annotate(ax=ax,
                           text='$z_{{{i}}}$'.format(i=idx),
                           xy=x,
                           xytext=(x.real, x.imag),
                           alpha=alpha)
  
  ax = NR_complex_annotate(ax=ax,
                           text='$\mathcal{{P}} \\left( z_{{{i}}} \\right)$'.format(i=idx),
                           xy=x,
                           xytext=(P.f(x).real, P.f(x).imag),
                           alpha=alpha)
  
  return ax

def NR_complex_method_ax(ax, P, x_0, idx=1, alpha=1.0,
                         title=None):
  
  if idx > 0:
    idx_old = idx - 1
    alpha_old = 0.2
  else:
    idx_old = idx
    alpha_old = 1.0
  
  ax = NR_complex_points(ax, P, x_0, idx=idx_old, alpha=alpha_old)
  
  ax.set_title(title, fontsize=20, fontweight='bold')

  if idx > 0:
    x_1 = NR_step(P, x_0)
  
    ax = NR_complex_points(ax, P, x_1, idx=idx, alpha=alpha)

    # Draw arrow between old z_{n-1} and new z_{n} points
    l = x_1 - x_0
    arrow_s = x_0+l/10
    arrow_e = l-3*l/10
    ax.arrow(arrow_s.real, arrow_s.imag,
             arrow_e.real, arrow_e.imag,
             lw=3, width=0.01)
    # Draw arrow between old P(z_{n-1}) and new P(z_{n}) points
    l = P.f(x_1) - P.f(x_0)
    arrow_s = P.f(x_0)+l/10
    arrow_e = l-2.5*l/10
    ax.arrow(arrow_s.real, arrow_s.imag,
             arrow_e.real, arrow_e.imag,
             lw=3, width=0.01)
    return ax, x_1

  return ax

def NR_complex_method(P,
                      x_0=(0.47+0.56j),
                      grid_lim=None):
  
  if grid_lim is None:
    grid_lim = NR_missing_grid_lim(P)
  
  fig, axes = NR_complex_fig_setup(nrows=2, ncols=2,
                                   grid_lim=grid_lim, axis=True)

  for i, ax in enumerate(axes):
    if i != 0:
      alpha = 0.3
    else:
      alpha = 0.8
    ax, _ = NR_intro_roots(ax, P, alpha=alpha)

  _ = NR_complex_method_ax(axes[0], P=P, x_0=x_0, idx=0,
                          title=('0. lépés: A kezdőpont ($\mathbf{z_{0}}$) és a hozzá\n' +
                                 'tartozó függvényérték ($\mathbf{P \\left( z_{0}' +
                                 '\\right)}$) felvétele')
                         )
  _, x_1 = NR_complex_method_ax(axes[1], P=P, x_0=x_0, idx=1,
                          title=('1. lépés: A kezdőpont léptetése az NR-módszer\n' +
                                 'segítségével: '+
                                 '$\mathbf{z_{n+1} = z_{n} - '+
                                 'P \\left( z_{n} \\right) / '+
                                 'P\,\' \\left( z_{n} \\right)}$')
                         )
  _, x_2 = NR_complex_method_ax(axes[2], P=P, x_0=x_1, idx=2,
                          title=('2. lépés: Az előző lépés ismétlése')
                         )
  _, x_3 = NR_complex_method_ax(axes[3], P=P, x_0=x_2, idx=3,
                          title=('További lépések: Ismétlés addig, míg az érték\n'+
                                 'be nem konvergál az egyik zérushelyre')
                         )
    
  fig.suptitle('Fig. 5. Newton--Raphson-módszer a komplex síkon',
               fontsize=20, fontweight='bold',
               y=0.07)
    
  plt.show()



#######
#
#    NEWTON-FRACTAL STEPS FOR 3 DIFFERENT VIEWS
#
##########################################################################


#
#  Function to display scatter points
#
def NR_fractal_ax(ax, X,
                  P=None, X_c=None):

  if (P is not None) & (X_c is not None):
    colors = get_NR_colors(P, X_c)
  else:
    colors = 'gray'
    
  ax.scatter(X.real, X.imag,
             c=colors, s=5**2, alpha=0.6)
  
  return ax

def NR_fractal_steps_gray(P, N=10,
                          steps=[1,3,6,10,15],
                          grid_lim=None):
  
  if grid_lim is None:
    grid_lim = NR_missing_grid_lim(P)
  
  fig, axes = NR_complex_fig_setup(nrows=2, ncols=3,
                                   grid_lim=grid_lim, axis=False)
  for ax in axes:
    ax, _ = NR_colored_roots(ax, P, alpha=0.8)

  # Plot NR steps
  X_0 = get_starting_grid(N, grid_lim, grid_lim)
  _ = NR_fractal_ax(axes[0], X=X_0)
  
  for i in range(1, len(axes)):
    X_1 = NR_iter(P, X_0, steps[i-1])
    _ = NR_fractal_ax(axes[i], X=X_1)
    X_0 = X_1
    
  fig.suptitle('Fig. 6. Newton--Raphson-módszer egy pontráccsal',
               fontsize=20, fontweight='bold',
               y=0.07)
    
  plt.show()

def NR_fractal_steps_colored(P, N=10,
                             steps=[1,3,6,10,15],
                             grid_lim=None):
  
  if grid_lim is None:
    grid_lim = NR_missing_grid_lim(P)
  
  fig, axes = NR_complex_fig_setup(nrows=2, ncols=3,
                                   grid_lim=grid_lim, axis=False)
  for ax in axes:
    ax, _ = NR_colored_roots(ax, P, alpha=0.8)

  # Plot NR steps
  X_0 = get_starting_grid(N, grid_lim, grid_lim)
  _ = NR_fractal_ax(axes[0], X=X_0, P=P, X_c=X_0)
  
  for i in range(1, len(axes)):
    X_1 = NR_iter(P, X_0, steps[i-1])
    _ = NR_fractal_ax(axes[i], X=X_1, P=P, X_c=X_1)
    X_0 = X_1
    
  fig.suptitle('Fig. 7. Newton--Raphson-módszer színezett pontráccsal',
               fontsize=20, fontweight='bold',
               y=0.07)
    
  plt.show()
  
def NR_fractal_steps_reversed(P, N=100,
                              steps=[1,3,6,10,15],
                              grid_lim=None):
  
  if grid_lim is None:
    grid_lim = NR_missing_grid_lim(P)
  
  fig, axes = NR_complex_fig_setup(nrows=2, ncols=3,
                                   grid_lim=grid_lim, axis=False)
  
  # Plot NR steps with scatter points
  X_0 = get_starting_grid(N, grid_lim, grid_lim)
  ax = NR_fractal_ax(axes[0], X=X_0, P=P, X_c=X_0)
  
  for i in range(1, len(axes)):
    X_N = NR_iter(P, X_0, steps[i-1])
    ax = NR_fractal_ax(axes[i], X=X_0, P=P, X_c=X_N)
    
  fig.suptitle('Fig. 8. A Newton--Raphson-fraktál pontrácson',
               fontsize=20, fontweight='bold',
               y=0.07)
  
  plt.show()


def NR_fractal_image_ax(ax, P, X_c,
                        grid_lim_x, grid_lim_y):

  colors = get_NR_colors(P, X_c)
  X = np.array(colors.reshape((int(np.sqrt(len(colors))),
                               int(np.sqrt(len(colors))),
                               4
                              ))
              )
  ax.imshow(X, extent=(*grid_lim_x, *grid_lim_y))
  
  return ax

def NR_fractal_steps_image(P, N=100,
                           steps=[1,3,6,10,15],
                           grid_lim=None):
  
  if grid_lim is None:
    grid_lim = NR_missing_grid_lim(P)
  
  fig, axes = NR_complex_fig_setup(nrows=2, ncols=3,
                                   grid_lim=grid_lim, axis=False)
  
  # Plot NR steps with scatter points
  X_0 = get_starting_grid(N, grid_lim, grid_lim)
  ax = NR_fractal_image_ax(ax=axes[0], P=P, X_c=X_0,
                           grid_lim_x=grid_lim, grid_lim_y=grid_lim)
  
  for i in range(1, len(axes)):
    X_N = NR_iter(P, X_0, steps[i-1])
    ax = NR_fractal_image_ax(axes[i], P=P, X_c=X_N,
                             grid_lim_x=grid_lim, grid_lim_y=grid_lim)
    
  fig.suptitle('Fig. 9. A Newton--Raphson-fraktál pixelekkel',
               fontsize=20, fontweight='bold',
               y=0.07)
  
  plt.show()


def NR_fractal_plot(P, N=512, n_steps=10,
                    grid_lim=None):

  fig, axes = NR_complex_fig_setup(nrows=1, ncols=1,
                                   grid_lim=grid_lim, axis=False)
  
  X_0 = get_starting_grid(N, grid_lim, grid_lim)
  X_N = NR_iter(P, X_0, n_steps)
  ax = NR_fractal_image_ax(ax=axes[0], P=P, X_c=X_N,
                           grid_lim_x=grid_lim, grid_lim_y=grid_lim)

  fig.suptitle('Fig. 10. Egy Newton--Raphson-fraktál',
               fontsize=20, fontweight='bold',
               y=0.02)
  
  plt.show()
  
def NR_fractal(P,
               N=150, n_steps=10, figsize=(10,10),
               grid_lim_x=None,
               grid_lim_y=None,
               axis=True, show=True,
               save=False, savedir='./out/'):

  if grid_lim_x is None: grid_lim_x = NR_missing_grid_lim(P)
  if grid_lim_y is None: grid_lim_y = NR_missing_grid_lim(P)
  
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
  ax.set_aspect('equal')
  if not axis:
    ax.axis('off')
  
  X_0 = get_starting_grid(N, grid_lim_x, grid_lim_y[::-1])
  X_N = NR_iter(P, X_0, n_steps)
  ax = NR_fractal_image_ax(ax=ax, P=P, X_c=X_N,
                           grid_lim_x=grid_lim_x, grid_lim_y=grid_lim_y)
  
  fname = 'nrfractal|N{0}|ns{1}|x{2}_{3}|y{4}_{5}.'.format(N, n_steps,
                                                           *grid_lim_x,
                                                           *grid_lim_y)
  if save:
    if not os.path.exists(savedir):
      os.makedirs(savedir)

    plt.savefig(savedir + fname + figsave_fmt,
                format=figsave_fmt,
                dpi=figsave_dpi,
                bbox_inches='tight')
  
  if show:
    plt.show()
  else:
    plt.clf()
    plt.close('all')



#######
#
#    OTHER FUNCTIONS
#
##########################################################################

def is_square_plot(v_coords):

  fig, ax = plt.subplots(figsize=(15,15))
  ax.set_aspect('equal')

  l = v_coords[0] - v_coords[-1]
  v_s = v_coords[0]
  for x, y, dx, dy in zip(v_s[:,0],v_s[:,1], l[:,0],l[:,1]):
    ax.arrow(x, y, -dx, -dy)
  for c in frame:
    ax.scatter(c[:,0], c[:,1], s=5**2, c='tab:red')

    wh = c[3] - c[0]
    rect = Rectangle(c[0], wh[0], wh[1],
                     lw=2, alpha=0.3, color='tab:green', fc='none')
    ax.add_patch(rect)

  plt.show()