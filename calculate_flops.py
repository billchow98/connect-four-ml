# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

from common import *
from network import summarize_model

# nn.tabulate is currently unreliable. FLOPs counting is broken on some devices.
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    summarize_model()  # Print model summary to debug output
