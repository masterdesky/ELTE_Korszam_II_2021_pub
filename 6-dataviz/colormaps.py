import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict


def puc_lightness():
  
  cmaps = OrderedDict()
  
  cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']

  # Number of colormap per subplot for particular cmap categories
  _DSUBS = {'Perceptually Uniform Sequential': 5}

  # Spacing between the colormaps of a subplot
  _DC = {'Perceptually Uniform Sequential': 1.4}
  
  # Indices to step through colormap
  x = np.linspace(0.0, 1.0, 100)

  # Do plot
  for cmap_category, cmap_list in cmaps.items():

      # Do subplots so that colormaps have enough space.
      # Default is 6 colormaps per subplot.
      dsub = _DSUBS.get(cmap_category, 6)
      nsubplots = int(np.ceil(len(cmap_list) / dsub))

      # squeeze=False to handle similarly the case of a single subplot
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18,5),
                              facecolor='0.95', squeeze=False)

      for i, ax in enumerate(axs.flat):

          locs = []  # locations for text labels

          for j, cmap in enumerate(cmap_list[i*dsub:(i+1)*dsub]):

              # Get RGB values for colormap and convert the colormap in
              # CAM02-UCS colorspace.  lab[0, :, 0] is the lightness.
              rgb = cm.get_cmap(cmap)(x)[np.newaxis, :, :3]
              lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

              # Plot colormap L values.  Do separately for each category
              # so each plot can be pretty.  To make scatter markers change
              # color along plot:
              # http://stackoverflow.com/questions/8202605/
              y_ = lab[0, :, 0]
              c_ = x

              dc = _DC.get(cmap_category, 1.4)  # cmaps horizontal spacing
              ax.scatter(x + j*dc, y_, c=c_, cmap=cmap, s=30**2, linewidths=0.0)

              # Store locations for colormap labels
              locs.append(x[-1] + j*dc)

          # Set up the axis limits:
          #   * the 1st subplot is used as a reference for the x-axis limits
          #   * lightness values goes from 0 to 100 (y-axis limits)
          ax.set_xlim(axs[0, 0].get_xlim())
          ax.set_ylim(0.0, 100.0)

          # Set up labels for colormaps
          ax.xaxis.set_ticks_position('top')
          ticker = mpl.ticker.FixedLocator(locs)
          ax.xaxis.set_major_locator(ticker)
          formatter = mpl.ticker.FixedFormatter(cmap_list[i*dsub:(i+1)*dsub])
          ax.xaxis.set_major_formatter(formatter)
          ax.xaxis.set_tick_params(rotation=50)
          
          ax.tick_params(axis='both', which='both', labelsize=20)
          
          ax.set_ylabel('Lightness $L^{*}$',
                        fontsize=25, fontweight='bold')

      ax.set_xlabel(cmap_category + ' colormaps',
                    fontsize=25, fontweight='bold', labelpad=20)

      fig.tight_layout(h_pad=0.0, pad=1.5)
      plt.show()