## https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
# We need this to fixate the limits of plots
def fixlims(obj):
    obj.xlim(obj.gca().get_xlim())
    obj.ylim(obj.gca().get_ylim())
