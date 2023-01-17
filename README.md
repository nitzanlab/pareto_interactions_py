# Pareto Interactions (python)

Both this repository and its sister, Mathematica repository [pareto_interactions_m](xx), supplement our preprint: [Emergence of division of labor in tissues through cell interactions and spatial cues](https://www.biorxiv.org/content/10.1101/2022.11.16.516540v1).

In our work, we:

1. develop a computational framework to investigate the contribution of global gradients across the tissue and of self-organization
mechanisms towards the division of labor within a cell-type population. 

2. identify distinguishable expression patterns emerging from cell-cell interactions vs. instructive signals.

3. propose a method to construct ligand-receptor networks between specialist cells and use it to infer division-of-labor mechanisms from single-cell RNA-seq and spatial transcriptomics data of stromal, epithelial, and immune cells.

This repository holds mostly variations of the simulation framework (supp figs 1-3).
<!-- , described briefly below.  -->
In addition, given the spatial and task distances, this repository generates the binned task distances vs physical distances plots (e.g., fig3. C, [example](xx)) and [maps](xx) and [explores](xx) the colon fibroblast single-cell RNA-seq data projection onto the Slide-seq data (fig4. D,f, and supp fig 5).

[Pareto_interactions_m](xx) contains the simulation framework as well (e.g., fig2.B), the data analysis of expression data (e.g., fig3. K) and archetype crosstalk method (e.g., fig4. C).

<!-- ## Pareto optimality framework with a cell-cell communication mechanism

Pareto-opmality theory predicts that the optimal performance of a multitasker cell that faces
trade-offs (e.g., due to finite resources (Sabi and Tuller 2019; Shoval et al. 2012)) is achieved when
its expression is bounded inside a polytope whose vertices are expression profiles optimal at each
task, called archetypes (Shoval et al. 2012; Korem et al. 2015; Hart et al. 2015; Hausser et al. 2019). 
The Pareto optimality theory was recently extended to consider an ensemble of cells
that are working as a collective to perform the tissue’s tasks (Adler et al. 2019). 


To model the Pareto-optimal expression profiles of cells in a tissue, we consider how cells
collectively contribute to the tissue by performing several tasks. As was previously presented
(Shoval et al. 2012; Adler et al. 2019), we model this trade-off by considering that each task is best
performed at an optimal expression profile, �!
∗
, (or an optimal task allocation) and shows a decline
in performance as cells move further away from �!
∗ in gene expression space. We define the total
performance function of a tissue, �, as a product over the performance of all the tasks that need to
be collectively performed by the cells in the tissue, summing over the contribution of each cell to
the performance in each task (Adler et al. 2019) (Methods).
To model the effect of cell-cell interactions on optimal task allocation, we introduce an interaction
term, �!, which captures how a cell’s performance is influenced by the performance of its
neighboring cells. We explore the effect of varying the range of the interaction by varying the size
of the neighborhood of each cell (�#). The contribution of each cell (�) in task � is therefore the
product of two components; a self-component, �!, which is a function of cell �’s gene expression
profile (�#), and an interaction component, �!, which is a function of the average �! of the
neighboring cells of cell � (Figure 2A).
The interaction term, �!, can generally represent different types of interactions, including positive
and negative effects on performing the same task. Here, we focus on lateral inhibition, where a
cell’s performance in task � declines if its neighboring cells exhibit high performance in the same
task (Figure 2A, Methods). We consider a representative example of a 2D spatial grid of 100 cells
that need to perform three tasks and compute the expression profiles (or task allocations) of the
cells that maximize � (Methods). We discuss tissue dimensions, number of tasks, and other types
of interactions in the supplementary information (SI) (Figure S1-2).


We optimize the collective cellular task performance under trade-offs, we find that distinguishable expression patterns can emerge from cell-cell interactions vs. instructive signals. We propose a method to construct ligand-receptor networks between specialist cells and use it to infer division-of-labor mechanisms from single-cell RNA-seq and spatial transcriptomics data of stromal, epithelial, and immune cells. Our framework can be used to characterize the complexity of cell interactions within tissues.
 -->


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
