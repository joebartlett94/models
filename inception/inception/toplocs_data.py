"""Small library that points to the toplocs data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from inception.dataset import Dataset


class TopLocsData(Dataset):
  """Top Locs data set."""

  def __init__(self, subset):
    super(TopLocsData, self).__init__('TopLocs', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 10

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 4741
    if self.subset == 'validation':
      return 1184

  def download_message(self):
    """N/A"""
    print('N/A for TopLocs dataset')