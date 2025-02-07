#!/usr/bin/env python

import os
import itertools
import numpy as np
from multicorner import mcorner
import matplotlib
matplotlib.use('agg')
import random
import matplotlib.pyplot as plt


data = np.load('data.npz')
samples1 = data['dataset']
labels = data['labels']
n_clusters = data['n']
indices = data['inds']


samples2 = [samples1[labels == i] for i in range(n_clusters)] #manually clustered


# Parameter options
autocluster_options = [True, False]
confidence_ellipse_options = [True, False]
nsigma_options_autocluster_true = [1, [1, 2, 3, 4, 5]]
nsigma_options_autocluster_false = [1, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
truths_options = [None, [1, 2, 3]]
upper_title_options = [None, lambda i, j: f"Orientation Plots ({i},{j})"]
diag_title_options = [None, lambda i, j, ii, jj: f"Scatter Plots ({i},{j}) {ii},{jj}"]
lower_title_options = [None, lambda i, j, ii, jj: f"Histograms ({i},{j}) {ii},{jj}"]
percentile_bounds_options = [[16, 84], [2.5, 97.5]]
percentiles_options = [True, False]
labels_options = [None, ['x', 'y', 'z']]
parameter_combinations = list(itertools.product(
    autocluster_options, confidence_ellipse_options, truths_options,
    upper_title_options, diag_title_options, lower_title_options,
    percentile_bounds_options, percentiles_options, labels_options))



#print(len(parameter_combinations))


path= './testplots/'
try:
    os.makedirs(path)
except FileExistsError:
    pass


# Loop over only the pre-selected indices
for index in indices:
    autocluster, confidence_ellipse, truths, upper_title, diag_title, lower_title, percentile_bounds, percentiles, labels = parameter_combinations[index]
    
    # Determine which sample set to use
    samples = samples1 if autocluster else samples2
    nsigma_options = nsigma_options_autocluster_true if autocluster else nsigma_options_autocluster_false
    
    for nsigma in nsigma_options:
        # Call the function
        fig = mcorner(
            samples=samples,
            autocluster=autocluster,
            confidence_ellipse=confidence_ellipse,
            nsigma=nsigma,
            truths=truths,
            upper_title=upper_title,
            diag_title=diag_title,
            lower_title=lower_title,
            percentile_bounds=percentile_bounds,
            percentiles=percentiles,
            verbose=False,
            labels=labels,
            fontsize=6,
            labelsize=6,
            outer_wspace=0.2,
            outer_hspace=0.2,
            inner_wspace=0.2,
            inner_hspace=0.1,
            figsize=25
        )

        # Save the figure with an indexed filename
        fig.savefig(path + f"test_plot_{index + 1}.png",bbox_inches='tight')
        plt.close(fig)


#Now compare all the images to ensure they're close

import pytest
import os
from PIL import Image
import numpy as np

# Directories
OUTPUT_DIR = "testplots"
GROUND_TRUTH_DIR = "ground_truth"

# List PNG files in both directories
output_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png"))
ground_truth_files = sorted(f for f in os.listdir(GROUND_TRUTH_DIR) if f.endswith(".png"))

# Ensure both directories have the same filenames
assert output_files == ground_truth_files, "File lists do not match between output and ground truth."

def mse(image1, image2):
    """Compute Mean Squared Error (MSE) between two images."""
    img1 = np.array(image1)
    img2 = np.array(image2)

    # Ensure images have the same shape
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    # Compute MSE
    diff = np.mean((img1 - img2) ** 2)
    return diff

@pytest.mark.parametrize("filename", output_files)
def test_images_mse(filename):
    """Compare output images with ground truth using MSE."""
    output_path = os.path.join(OUTPUT_DIR, filename)
    ground_truth_path = os.path.join(GROUND_TRUTH_DIR, filename)

    img1 = Image.open(output_path).convert("RGB")  # Convert to RGB
    img2 = Image.open(ground_truth_path).convert("RGB")

    mse_value = mse(img1, img2)

    assert mse_value < 2.0, f"Image {filename} differs too much from ground truth (MSE = {mse_value:.4f})"

