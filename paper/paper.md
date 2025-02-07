# multicorner.py: pairplots for multi-modal distributions

**Zachary Murray**  
*Research Excellence Fellow, Université Côte d'Azur*  
**Date:** \\( \today \\)

---

**Software Repository:** <https://github.com/dfm/corner.py>  
<!--
**Paper DOI:** <INSERT if Accepted>  
**Software Archive:** <INSERT if Accepted>
-->

---

## Summary

Pairplots (also known as *Corner Plots* and *Scatterplot matrices*) are one of the best ways of visualizing high-dimensional data. Starting with *corner.py* (originally *triangle*) by Dan Foreman-Mackey ([Foreman-Mackey 2016][1]), corner plots have been used extensively in many fields of the physical and computer sciences for visualizing high-dimensional data, especially the output of MCMC chains and other sampling methods [[1]](#ref-corner).

The popularity of corner plots has led to the development of several similar packages, each specializing in creating corner plots for one application or another. For example, [ChainConsumer][5] specializes in processing EMCEE or other sampling-method outputs (with advantages like handling multimodal posteriors), while Seaborn also includes pairplot capabilities for machine learning and statistical data visualization [[7]](#ref-seaborn). Julia users can use [PairPlots.jl][6].  

Other packages, like [cornerhex][8], improve upon the same concept by representing the density distribution with hexagonal, rather than square, bins. Hexagonal bins—being more circular—result in greater packing efficiency and less distortion [[9]](#ref-carr1987).  

However, whether applied to MCMC samples or to multivariate data more generally, these approaches can be less effective when there is a **significant separation in scale** between individual modes of a posterior and the distances between them. Fortunately, this type of strong scale-separation is ideal for clustering algorithms. **Multicorner** is a highly customizable package that combines the visualization strengths of corner/pairplots with clustering algorithms to better visualize *highly multimodal and multidimensional* datasets.

## Statement of Need

Corner plots featuring widely separated yet compact distributions are relatively rare compared to single-modal cases, but they still occur frequently across many scientific fields. Such distributions often arise in periodic models when datasets lack sufficient information to constrain periodicity—for instance, in orbital parameter estimation [[Blunt et al. 2017][10]] or in determining asteroid pole orientations [[Magnusson 1986][11]]—but they are not exclusive to these use cases. They can also occur in non-periodic scenarios such as spectroscopy [[Damiano et al. 2023][12]].

More broadly, multimodal distributions may also result from underestimated error bars [[Hogg et al. 2010][13]], often due to unaccounted-for or underestimated systematic errors. As astronomical observations improve in precision (and random errors decrease), systematic uncertainties play an increasingly critical role in data analysis.

Furthermore, scale separation is a common phenomenon in large datasets. Tightly clustered points (e.g., within cities) may be separated by vast distances, coral reefs may be spread across large oceanic regions, and analogous scale-separated patterns appear in numerous other domains. Given the prevalence of such scale-separated distributions, a specialized visualization tool is urgently needed.

Below is an example of a visualization produced with *multicorner*:

![Multicorner plot](multicornerplot.png)

And for comparison, here is the equivalent visualization using *corner.py*:

![Corner plot](cornerplot.png)

## Reading the Plot

Plots produced by *multicorner* are divided into three main sections:

1. **Lower Triangle**  
   Much like a traditional corner plot, the lower triangle presents information about each distribution in a grid of subplots. These are equivalent to a corner plot for each individual *mode* within the dataset.

2. **Diagonal**  
   The diagonal displays histograms of each individual *mode* in the dataset, again mirroring the approach of a traditional corner plot.

3. **Upper Triangle**  
   The upper triangle depicts the relative positions of the *modes* within the dataset. It highlights large-scale separations and orientations of modes, in contrast to the lower triangle’s focus on local covariances and densities.

---

## References

<a id="ref-corner"></a>  
**[1]** Foreman-Mackey, Daniel. 2016. “corner.py: Scatterplot Matrices in Python.” *The Journal of Open Source Software* 1(2): 24.  
<https://doi.org/10.21105/joss.00024>

**[2]** Foreman-Mackey, Daniel. 2016. *Corner.py on Github*.  
<https://github.com/dfm/corner.py>

**[3]** *Zenodo Archive.* 2016. “Corner.py: Scatterplot Matrices in Python.”  
<http://dx.doi.org/10.5281/zenodo.53155> (doi:10.5281/zenodo.53155)

**[5]** Hinton, S. R. 2016. “ChainConsumer.” *The Journal of Open Source Software* 1: 00045.  
<https://doi.org/10.21105/joss.00045>

**[6]** Thompson, William. 2021. *PairPlots.jl: Beautiful and Flexible Visualizations of High Dimensional Data.*  
<https://github.com/sefffal/PairPlots.jl> (Accessed: 2025-01-12)

**[7]** Waskom, Michael. 2012. *Seaborn: statistical data visualization.*  
<https://github.com/mwaskom/seaborn> (Accessed: 2025-01-12)

**[8]** Stammler, Sebastian. 2023. *cornerhex.*  
<https://github.com/stammler/cornerhex> (Accessed: 2025-01-12)

<a id="ref-carr1987"></a>  
**[9]** Carr, D. B., Littlefield, R. J., Nicholson, W. L., & Littlefield, J. S. 1987. “Scatterplot Matrix Techniques for Large N.” *Journal of the American Statistical Association* 82(398): 424–436.  
<http://www.jstor.org/stable/2289444>

<a id="ref-blunt2017"></a>  
**[10]** Blunt, S., Nielsen, E. L., De Rosa, R. J., Konopacky, Q. M., Ryan, D., Wang, J. J., Pueyo, L., Rameau, J., Marois, C., Marchis, F., et al. 2017. “Orbits for the Impatient: A Bayesian Rejection-Sampling Method for Quickly Fitting the Orbits of Long-Period Exoplanets.” *The Astronomical Journal* 153(5): 229.  
<https://doi.org/10.3847/1538-3881/aa6930>

<a id="ref-magnusson1986"></a>  
**[11]** Magnusson, P. 1986. “Distribution of Spin Axes and Senses of Rotation for 20 Large Asteroids.” *Icarus* 68(1): 1–39.  
<https://doi.org/10.1016/0019-1035(86)90072-2>

<a id="ref-damiano2023"></a>  
**[12]** Damiano, M., Hu, R., & Mennesson, B. 2023. “Reflected Spectroscopy of Small Exoplanets. III. Probing the UV Band to Measure Biosignature Gases.” *The Astronomical Journal* 166(4): 157.  
<https://doi.org/10.3847/1538-3881/acefd3>

<a id="ref-hogg2010"></a>  
**[13]** Hogg, D. W., Bovy, J., & Lang, D. 2010. “Data Analysis Recipes: Fitting a Model to Data.” *arXiv e-prints.*  
<https://doi.org/10.48550/arXiv.1008.4686>

