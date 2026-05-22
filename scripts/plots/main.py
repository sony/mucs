import sys

import image_pairs
import dist_distrib
import image_changes

# Define global variables
cfg = {
    "path_out": "pointer_to/plots/",
    # "ext_out": "png",
    "ext_out": "pdf",
    "cm2i": 0.3937,
    "height": 4,
    "width": 6,
    "linewidth": 1,
    "labelsize": 5,
    "fontsize": 6,
}

# Call subscripts
# image_pairs.run(cfg)
dist_distrib.run(cfg)
# image_changes.run(cfg)
