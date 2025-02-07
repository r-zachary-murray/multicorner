import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import norm
from scipy.spatial import KDTree

from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def best_rectangle(N):
    """
    Find the dimensions of the most square-like rectangle (grid) that can fit at least N items.

    This function computes the dimensions (width and height) of a rectangle that:
    - Contains at least N cells.
    - Is as close to a square as possible by minimizing the absolute difference between width and height.
    - Allows for some unused cells as long as the total number of cells is no less than N.

    Parameters:
    ----------
    N : int
        The number of items to arrange in the rectangle.

    Returns:
    -------
    tuple
        A tuple (width, height) representing the dimensions of the best-fitting rectangle, 
        where `width` is the smaller dimension and `height` is the larger dimension.

    """
    # Initialize variables to track the best rectangle found
    best_width = 1
    best_height = N
    best_difference = N  # Start with a large difference as the worst case

    # Iterate over possible widths up to ceil(sqrt(N)) + 1
    for width in range(1, int(N**0.5) + 2):
        # Calculate the minimum height needed to fit N items with the current width
        height = (N + width - 1) // width  # Equivalent to math.ceil(N / width)
        
        # Calculate the difference between width and height
        difference = abs(height - width)

        # Update the best rectangle if it improves on the current best
        if width * height >= N and difference < best_difference:
            best_width = width
            best_height = height
            best_difference = difference

    return best_width, best_height

def find_elbow_kneedle(x, y):
    """
    Find the elbow point in a curve using the Kneedle algorithm.

    This function identifies the point of maximum curvature (elbow point) in 
    a dataset by measuring the distance from each point to the diagonal of 
    the normalized data.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the data points (e.g., indices or input values).
    y : array-like
        The y-coordinates of the data points (e.g., metric values).

    Returns
    -------
    elbow_x : float
        The x-coordinate of the detected elbow point.
    elbow_y : float
        The y-coordinate of the detected elbow point.
    """
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Normalize x and y
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Compute the distance from the diagonal line (y = x)
    distances = y_norm - x_norm

    # Find the index of the maximum distance
    elbow_index = np.argmax(np.abs(distances))

    # Return the corresponding x and y values
    elbow_x = x[elbow_index]
    elbow_y = y[elbow_index]

    return elbow_x, elbow_y

def get_elbow_KD(data,M=5):
    """
    Estimate the optimal distance threshold using the elbow method on k-nearest 
    neighbor distances.

    This function uses a KDTree to compute the distance to the Mth nearest 
    neighbor for all data points, sorts the distances, and then identifies 
    the "elbow" point in the sorted distances curve using the Kneedle algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D array where each row represents a data point in multi-dimensional 
        space.
    M : int, optional, default=5
        The number of neighbors to consider when calculating distances.

    Returns
    -------
    float
        The Mth nearest neighbor distance at the elbow point, which can be 
        used as a threshold for clustering or outlier detection.
    """
    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(data)
    # Compute the distance to the Mth nearest neighbor for all points
    distances, _ = tree.query(data, k=M)
    # Extract the Mth nearest neighbor distances (distances[:, M-1])
    mth_nearest_distances = distances[:, M-1]
    xs = np.arange(0,len(distances),1)
    ys = np.sort(mth_nearest_distances)
    xelb,yelb = find_elbow_kneedle(xs, ys)
    return yelb


def plot_confidence_ellipse(covariance_matrix, ax, mu, n_sigma=3, **kwargs):
    """
    Plot a confidence ellipse for a given 2D covariance matrix onto the specified axis.

    This function visualizes the uncertainty represented by a 2D covariance matrix 
    as an ellipse. The ellipse is defined based on the eigenvalues and eigenvectors 
    of the covariance matrix, scaled by the specified number of standard deviations (n_sigma).

    Parameters:
    ----------
    covariance_matrix : numpy.ndarray
        A 2x2 covariance matrix representing the variance and covariance of two variables.
    ax : matplotlib.axes.Axes
        The matplotlib axis object on which the ellipse will be drawn.
    mu : tuple or list of float
        The mean (center) of the ellipse in the form (x, y).
    n_sigma : float, optional, default=3
        The number of standard deviations to determine the size of the ellipse.
        For example, n_sigma=1 corresponds to ~68% confidence, n_sigma=2 to ~95%, and n_sigma=3 to ~99.7%.
    **kwargs : dict, optional
        Additional keyword arguments to customize the appearance of the ellipse, 
        such as `edgecolor`, `facecolor`, `alpha`, etc. These are passed directly 
        to the `matplotlib.patches.Ellipse` constructor.

    Returns:
    -------
    matplotlib.patches.Ellipse
        The matplotlib Ellipse object added to the axis.

    """
    # Decompose covariance matrix into eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvalues and eigenvectors in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Compute the angle of rotation of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Compute the width and height of the ellipse based on eigenvalues and n_sigma
    width, height = 2 * n_sigma * np.sqrt(eigenvalues)

    # Create the ellipse patch
    ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, **kwargs)

    # Add the ellipse to the provided axis
    ax.add_patch(ellipse)

    return ellipse


def fancyscatter(x, y):
    """
    Perform a fancy scatter plot analysis using Gaussian Kernel Density Estimation (KDE).

    This function computes the density of points in a 2D space and identifies regions
    corresponding to 1σ, 2σ, and 3σ confidence levels. It also separates points inside
    and outside the 1σ contour for further analysis or plotting.

    Parameters:
    ----------
    x : numpy.ndarray
        1D array of x-coordinates of the data points.
    y : numpy.ndarray
        1D array of y-coordinates of the data points.

    Returns:
    -------
    X : numpy.ndarray
        2D array of x-coordinates of the grid used for KDE evaluation.
    Y : numpy.ndarray
        2D array of y-coordinates of the grid used for KDE evaluation.
    Z : numpy.ndarray
        2D array of density values evaluated over the grid (same shape as X and Y).
    x_outside : numpy.ndarray
        1D array of x-coordinates of points outside the 1σ contour.
    y_outside : numpy.ndarray
        1D array of y-coordinates of points outside the 1σ contour.
    x_inside : numpy.ndarray
        1D array of x-coordinates of points inside the 1σ contour.
    y_inside : numpy.ndarray
        1D array of y-coordinates of points inside the 1σ contour.
    thresholds : list of float
        Density thresholds corresponding to 1σ, 2σ, and 3σ confidence levels
    """

    # Stack inputs for fitting (sklearn expects samples in rows)
    values = np.vstack([x, y]).T
    bw = min(np.ptp(x), np.ptp(y)) / 30
    # Create and fit KernelDensity model
    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde.fit(values)

    # Create a grid over which we’ll evaluate the KDE
    x_grid = np.linspace(x.min() - 1, x.max() + 1, 100)
    y_grid = np.linspace(y.min() - 1, y.max() + 1, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()]).T

    # Evaluate the log-density model on the grid and exponentiate
    log_density = kde.score_samples(positions)
    Z = np.exp(log_density).reshape(X.shape)

    # Compute density thresholds for n-sigma contours
    Z_flat = Z.ravel()
    sorted_Z = np.sort(Z_flat)
    cumulative_density = np.cumsum(sorted_Z) / np.sum(sorted_Z)

    # Map sigma levels to cumulative density (1σ, 2σ, 3σ)
    sigma_levels = [1 - np.exp(-n**2 / 2) for n in range(1, 4)]  
    thresholds = [sorted_Z[np.searchsorted(cumulative_density, level)] for level in sigma_levels]
    threshold_1sigma = thresholds[0]  # 1σ threshold

    # Evaluate densities for all original data points
    log_densities = kde.score_samples(values)  # log-density for each data point
    densities = np.exp(log_densities)

    # Filter points outside/inside 1σ contour
    outside_mask = densities < threshold_1sigma
    x_outside = x[outside_mask]
    y_outside = y[outside_mask]

    inside_mask = densities > threshold_1sigma
    x_inside = x[inside_mask]
    y_inside = y[inside_mask]

    return X, Y, Z, x_outside, y_outside, x_inside, y_inside, thresholds



def auto_clusters(samples,eps=None):
    """
    Perform automatic clustering of data points using the DBSCAN algorithm, 
    with handling for noise points by reassigning them to the nearest cluster.

    Parameters:
    ----------
    samples : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the data points 
        to be clustered. Each row corresponds to a data point, and each column 
        corresponds to a feature.
    eps : float, optional, default=0.5
        The maximum distance between two samples for them to be considered 
        as part of the same neighborhood in the DBSCAN algorithm.

    Returns:
    -------
    list of numpy.ndarray
        A list where each element is a 2D array containing the samples belonging 
        to one cluster. Noise points, if reassigned to clusters, will appear in 
        their respective new clusters.
    """
    #if  no eps is assinged, try to pick one based on elbow method (e.g. https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012)
    if eps == None:
        eps = 3*get_elbow_KD(samples,M=5)
    dbscan = DBSCAN(eps=eps, min_samples=10)
    dbscan.fit(samples)
    labels = dbscan.labels_

    # Perform DBSCAN clustering
    # Handle noise points (-1)
    if -1 in labels:
        noise_points = samples[labels == -1]
        cluster_points = samples[labels != -1]
        cluster_labels = labels[labels != -1]

        # Compute centroids of clusters
        unique_clusters = np.unique(cluster_labels)
        centroids = np.array([samples[labels == c].mean(axis=0) for c in unique_clusters])

        # Assign noise points to nearest cluster
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(centroids)
        nearest_cluster = nn.kneighbors(noise_points, return_distance=False).ravel()

        # Update labels
        for i, point in enumerate(noise_points):
            cluster_idx = nearest_cluster[i]
            labels[np.where((samples == point).all(axis=1))] = unique_clusters[cluster_idx]

    # Separate samples into different arrays based on updated labels
    unique_labels = set(labels)
    clustered_samples = [samples[labels == i] for i in unique_labels]

    return clustered_samples


def mcorner(samples, autocluster=True, confidence_ellipse=False, nsigma=3, truths=None, upper_title=None, diag_title=None, lower_title=None, percentile_bounds=[16, 84], percentiles=False, verbose=False, labels=None, fontsize=12,titlesize=10,labelsize=8,ticksize=6,outer_wspace=0.2,outer_hspace=0.2,inner_wspace=0.3,inner_hspace=0.35,figsize=None,eps=None,cmapname=None,binnum=30):
    """
    Generate a multivariate corner plot with clustering, confidence ellipses, 
    and customizable titles.

    This function creates a matrix of plots to visualize pairwise relationships 
    and distributions of multidimensional data. Optional features include:
    - Automatic clustering of the data.
    - Confidence ellipses for 2D subplots.
    - Customizable titles for subplots.

    Parameters
    ----------
    samples : list of numpy.ndarray
        A list of clusters, where each cluster is represented as a 2D array 
        of shape `(n_points, n_dimensions)`.
    autocluster : bool, optional, default=True
        Whether to automatically cluster the data. If True, the `auto_clusters` 
        function is used.
    confidence_ellipse : bool, optional, default=False
        Whether to draw confidence ellipses on the 2D scatter plots.
    nsigma : int, float, list, or numpy.ndarray, optional, default=3
        Sigma level(s) for the confidence ellipses. If a list or array is 
        provided, it must match the number of clusters.
    truths : iterable, optional, default=None
        Ground truth values for the dimensions. If provided, these are 
        indicated with markers and lines.
    upper_title : callable or None, optional, default=None
        Function to generate titles for subplots in the upper triangle. 
        The function should accept subplot indices `(i, j)` as arguments.
    diag_title : callable, str, or None, optional, default=None
        Function or string to generate titles for diagonal subplots. If 
        `"percentile"`, percentile bounds are shown.
    lower_title : callable or None, optional, default=None
        Function to generate titles for subplots in the lower triangle. 
        The function should accept subplot indices `(i, j, ii, jj)` as arguments.
    percentile_bounds : list or tuple of float, optional, default=[16, 84]
        Percentile bounds used for diagonal histograms. Defaults to the 68% 
        confidence interval.
    percentiles : bool, optional, default=False
        If True, vertical lines indicating percentile bounds are added to the 
        diagonal histograms.
    verbose : bool, optional, default=False
        If True, returns the clustered samples alongside the generated figure.
    labels : list of str or None, optional, default=None
        Axis labels for each dimension. The length must match the number of 
        dimensions.
    fontsize : int, optional, default=12
        Font size for subplot titles.
    titlesize : int, optional, default=10
        Font size for text inside subplots.
    labelsize : int, optional, default=6
        Font size for axis labels and tick labels.
    ticksize : int, optional, default=6
        Font size for tick labels.
    outer_wspace : float, optional, default=0.2
        Horizontal spacing between the subplots in the outer grid.
    outer_hspace : float, optional, default=0.2
        Vertical spacing between the subplots in the outer grid.
    inner_wspace : float, optional, default=0.3
        Horizontal spacing within each subplot's grid.
    inner_hspace : float, optional, default=0.35
        Vertical spacing within each subplot's grid.
    figsize : tuple or None, optional, default=None
        Figure size in inches `(width, height)`. If None, the size is 
        determined automatically based on the data dimensions and number 
        of clusters.
    eps : float or None, optional, default=None
        Epsilon parameter for the DBSCAN clustering algorithm. Automatically 
        computed if `autocluster` is True and `eps` is None.
    cmapname : str or None, optional, default=None
        Name of the colormap used to color the clusters. If None, an HSV-based 
        colormap is used.
    binnum : int, optional, default=30
        Number of bins for histograms on the diagonal plots.

    Returns
    -------
    matplotlib.figure.Figure
        The generated corner plot.
    list of numpy.ndarray, optional
        Clustered samples, returned only if `verbose=True`.
    """

    if autocluster and eps==None:
        clustered_samples = auto_clusters(samples)
    elif autocluster and eps != None:
        clustered_samples = auto_clusters(samples,eps)
    elif not autocluster:
        clustered_samples = samples
    else:
        raise ValueError("autocluster must be True or False")

    median_pos = np.array([np.median(sample, axis=0) for sample in clustered_samples])
    covs = np.array([np.cov(sample, rowvar=False) for sample in clustered_samples])

    N = clustered_samples[0].shape[-1]
    n_clusters = len(clustered_samples)
    M1, M2 = best_rectangle(n_clusters)

    if cmapname == None:
        colors = [hsv_to_rgb((i / n_clusters, 1, 1)) for i in range(n_clusters)]
    else:
        colormap = cm.get_cmap(cmapname, n_clusters)  # Replace 'viridis' with any desired colormap
        colors = [colormap(i / n_clusters) for i in range(n_clusters)]

    if confidence_ellipse not in [True, False]:
        raise ValueError("confidence_ellipse must be True or False")

    if not isinstance(nsigma, (int, float, list, np.ndarray)):
        raise TypeError("nsigma must be a float or iterable of floats, e.g., a list or numpy array")

    if isinstance(nsigma, (list, np.ndarray)) and len(nsigma) != len(clustered_samples):
        raise ValueError(f"If nsigma is iterable, it must have the same length as clustered_samples ({len(clustered_samples)})")

    if truths is not None and len(truths) != N:
        raise ValueError(f"truths must have the same number of dimensions as samples ({N})")

    if figsize == None:
        fig = plt.figure(figsize=(5 * N + np.sqrt(n_clusters), 5 * N + np.sqrt(n_clusters)))
    else:
        fig = plt.figure(figsize=(figsize,figsize))
    outer_gs = gridspec.GridSpec(N, N, figure=fig, wspace=outer_wspace, hspace=outer_hspace)

    for i in range(N):
        for j in range(N):
            truex, truey = (truths[j] if truths else None), (truths[i] if truths else None)

            if i < j:
                ax = fig.add_subplot(outer_gs[i, j])
                if upper_title:
                    ax.set_title(upper_title(i, j), fontsize=titlesize)
                ax.scatter(median_pos[:, j], median_pos[:, i], c=colors)
                if truths:
                    ax.scatter(truex, truey, marker='s', c='k')

                if confidence_ellipse:
                    nsigma_list = nsigma if isinstance(nsigma, (list, np.ndarray)) else [nsigma] * len(covs)
                    for q, (subcov, n_sigma) in enumerate(zip(covs, nsigma_list)):
                        plot_confidence_ellipse(subcov[np.ix_([i, j], [i, j])], ax, [median_pos[q, j], median_pos[q, i]], n_sigma=n_sigma, edgecolor=colors[q], facecolor='None')

                if labels != None:
                    ax.set_xlabel(labels[j], fontsize=labelsize)
                    ax.set_ylabel(labels[i], fontsize=labelsize)

                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
                ax.axvline(truex, c='k') if truex else None
                ax.axhline(truey, c='k') if truey else None
                ax.set_aspect(x_span / y_span if y_span else 1, adjustable='box')
                ax.tick_params(labelsize=ticksize)

            else:
                inner_gs = gridspec.GridSpecFromSubplotSpec(M1, M2, subplot_spec=outer_gs[i, j], wspace=inner_wspace, hspace=inner_hspace)
                for ii in range(M1):
                    for jj in range(M2):
                        ind = jj + ii * M2
                        if ind < len(clustered_samples):
                            ax_sub = fig.add_subplot(inner_gs[ii, jj])
                            subsamples = clustered_samples[ind]

                            if i != j:
                                X, Y, Z, x_outside, y_outside, x_inside, y_inside, thresholds = fancyscatter(subsamples[:, j], subsamples[:, i])
                                trans_cmap = LinearSegmentedColormap.from_list("tmap", [(0, list(colors[ind]) + [0]), (1, list(colors[ind]) + [1])])
                                ax_sub.scatter(x_outside, y_outside, color=colors[ind], s=0.1)
                                ax_sub.contour(X, Y, Z, levels=thresholds, colors='k', linewidths=1.5, linestyles='dashed')
                                ax_sub.hexbin(x_inside, y_inside, gridsize=10, mincnt=1, cmap=trans_cmap, edgecolor=None)
                                if labels != None:
                                    ax_sub.set_xlabel(labels[j], fontsize=labelsize)
                                    ax_sub.set_ylabel(labels[i], fontsize=labelsize)
                                xlim, ylim = ax_sub.get_xlim(), ax_sub.get_ylim()
                                if lower_title:
                                    ax_sub.set_title(lower_title(i, j, ii, jj), fontsize=titlesize)
                                if truths and (xlim[0] < truex < xlim[1]) and (ylim[0] < truey < ylim[1]):
                                    if i !=j:
                                        ax_sub.scatter(truex,truey,marker='s',c='k')
                                        ax_sub.axvline(truex,c='k')
                                        ax_sub.axhline(truey,c='k')
                            else:
                                pbounds = np.percentile(subsamples[:, i], percentile_bounds)
                                p50 = np.percentile(subsamples[:, i], 50)
                                ax_sub.hist(subsamples[:, i], color=colors[ind], histtype='step', bins=binnum)
                                ax_sub.set_yticks([])
                                if labels != None:
                                    ax_sub.set_xlabel(labels[j], fontsize=labelsize)
                                xlim, ylim = ax_sub.get_xlim(), ax_sub.get_ylim()
                                if diag_title == 'percentile':
                                    title = fr'${np.round(p50, 2)}^{{{np.round(pbounds[0] - p50, 2)}}}_{{{np.round(pbounds[1] - p50, 2)}}}$'
                                elif diag_title == None:
                                    title = None
                                else:
                                    title = diag_title(i, j, ii, jj)
                                ax_sub.set_title(title, fontsize=titlesize)
                                if percentiles:
                                    for p in pbounds:
                                        ax_sub.axvline(p, c='k', ls='--')
                                if truths and (xlim[0] < truex < xlim[1]):
                                    ax_sub.axvline(truey,c='k')

                            xlim, ylim = ax_sub.get_xlim(), ax_sub.get_ylim()
                            x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
                            ax_sub.set_aspect(x_span / y_span if y_span else 1, adjustable='box')
                            ax_sub.tick_params(labelsize=ticksize)

    return (fig, clustered_samples) if verbose else fig



