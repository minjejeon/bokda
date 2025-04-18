## https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
# We need this to fixate the limits of plots

#def fix_lims(self):
#    self.xlim(self.gca().get_xlim())
#    self.ylim(self.gca().get_ylim())

#def patch_matplotlib():
#    import matplotlib.pyplot as plt
#    plt.fix_lims = fix_lims.__get__(plt)

import matplotlib.pyplot as plt

def fix_lims(ax=None, x=True, y=True):
    ax = ax or plt.gca()
    if x: ax.set_xlim(ax.get_xlim())
    if y: ax.set_ylim(ax.get_ylim())

plt.fix_lims = fix_lims
