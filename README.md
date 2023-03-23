# Pareto Interactions (python)

Both this repository and its sister, Mathematica repository [pareto_interactions](https://github.com/miriadler/pareto_interactions), supplement our [preprint](https://www.biorxiv.org/content/10.1101/2022.11.16.516540v1): Emergence of division of labor in tissues through cell interactions and spatial cues.

<!-- ![graphical abstract](https://user-images.githubusercontent.com/20613396/227188474-6561dd9a-9bcf-460a-9fd2-0918cc2cff07.jpg) -->
<p align="center">
  <img src="https://user-images.githubusercontent.com/20613396/227188474-6561dd9a-9bcf-460a-9fd2-0918cc2cff07.jpg" width="470" height="400">
</p>
In our work, we:

1. develop a computational framework to investigate the contribution of global gradients across the tissue and of self-organization
mechanisms towards the division of labor within a cell-type population. 

2. identify distinguishable expression patterns emerging from cell-cell interactions vs. instructive signals.

3. propose a method to construct ligand-receptor networks between specialist cells and use it to infer division-of-labor mechanisms from single-cell RNA-seq and spatial transcriptomics data of stromal, epithelial, and immune cells.

## This repository

This repository contains:

1. Simulated optimization of Pareto-optimal task and spatial distributions ([Figure S1](https://github.com/nitzanlab/pareto_interactions_py/notebooks/sfig1.ipynb),[Figure S2](https://github.com/nitzanlab/pareto_interactions_py/notebooks/sfig2.ipynb),[Figure S3](https://github.com/nitzanlab/pareto_interactions_py/notebooks/sfig3.ipynb)).

2. Plotting and statistical significance testing of task distances verses physical distances ([Figure 3 C,F,J,M, Figure S4 C](https://github.com/nitzanlab/pareto_interactions_py/notebooks/fig3.ipynb))

3. Analysis of colon fibroblast Slide-seq data (Avraham-Davidi et al.) (Supplementary figure [5](https://github.com/nitzanlab/pareto_interactions_py/notebooks/sfig5.ipynb)).

4. Projection and analysis of colon fibroblast single-cell RNA-seq expression (Muhl et al.) onto Slide-seq data (Avraham-Davidi et al.) ([Figure 4 E-F](https://github.com/nitzanlab/pareto_interactions_py/notebooks/fig4.ipynb)).


![pareto_ex 001](https://user-images.githubusercontent.com/20613396/227187936-b09c08e1-9930-430a-b8bb-f331c067cb7a.jpeg)

<!-- Mapping and analysis of Slide-seq data projected onto the single-cell data -->
<!-- , described briefly below.  -->
<!-- In addition, given the spatial and task distances, this repository generates the binned task distances vs physical distances plots (e.g., fig3. C, [example](xx)) and [maps](xx) and [explores](xx) the colon fibroblast single-cell RNA-seq data projection onto the Slide-seq data (fig4. D,f, and supp fig 5). -->

[Pareto_interactions](https://github.com/miriadler/pareto_interactions) contains:

1. Mathematica implementation of simulations of Pareto-optimal task and spatial distributions (Figure 2, Figure 3A-B,D-E).

2. Task analysis of intestine enterocyte LCM data (Moor et al., Figure 3 H-I)

3. Task analysis of colon fibroblasts Slide-seq data (Avraham-Davidi et al., Figure 3 K-L, Figure S4, S6)

4. Task analysis of colon fibroblasts single-cell data (Muhl et al. Figure 4 B)

5. Task analysis of lung fibroblasts and macrophages (Adams et al., Figure 5 A,C)

5. Ligand-receptor-based archetype crosstalk networks (Figure 4 A,C, Figure 5 B,D,F, Figure S7, Figure S8)



## How to install pareto_interactions_py

Create a clean environment with `conda` using the `environment.yml` file from the this repository:

```
conda env create -f "https://raw.githubusercontent.com/nitzanlab/pareto_interactions_py/master/environment.yml"
```
(For older versions of `conda` one needs to download the `environment.yml` and use the local file for installation.)

For install in an already existing environment, use `pip` of the latest release from github:

```
pip install pareto_interactions_py@git+https://github.com/nitzanlab/pareto_interactions_py.git
```


## Contact

Feel free to contact us by [mail][email].

[email]: mailto:noa.moriel@mail.huji.ac.il