## 1 Introduction

Over the past few years, deep learning research has seen significant progress in solving graph learning tasks. A crucial aspect of such problems is maintaining equivariance to transformations, such as rotations and translations, allowing for instance, to reliably predict physical properties of molecules. In this section, we will provide an overview of the predominant methods used in this domain, along with an introduction to a new method: E(n) Equivariant Simplicial Message Passing Networks (EMPSNs) [3].

Graph Neural Networks (GNNs) [13], namely their most common variant, Message Passing Neural Networks (MPNNs) [5] are instrumental for learning on graph data. Simple MPNNs however, have a number of drawbacks. Firstly, they are limited in their ability to learn higher-dimensional graph structures such as cliques (a set of points that are all connected to each other), since communication normally only happens from nodes to other nodes. Secondly, they suffer from over-smoothing; nodes of a graph iteratively update their features by aggregating the features of their neighbors, a process by which updated node features become increasingly similar. Previous works attempt to improve MPSNs’ expressivity by considering higher-dimensional simplices in the graph as learnable features [1] [8]. While these methods provide the tools for more powerful graph learning models, they do not concern themselves with equivariance.

As stated in the original EMPSN [3] paper, many real-life problems have a natural symmetry to translations, rotations, and reflections (that is, to the Euclidean group E(n)), such as object recognition or predicting molecular properties. Many approaches have been proposed to ensure E(n) equivariance: Tensor field networks [14], SE(3) Transformers [4], E(n) Equivariant Graph Neural Networks [12] among others. These works are particularly useful for working with geometric graph data, such as molecular point clouds; they use the underlying geometry of the space in which the graph is positioned to ensure E(n) equivariance. In this case, however, the lack of higher-dimensional features remains a limiting factor for the reasons stated previously. EMPSNs [3] are a novel approach to learning on geometric graphs and point clouds that is equivariant to the euclidean group E(n) (rotations, translations, and reflections). The method combines geometric and topological graph approaches to leverage both benefits. Its main contributions related to our reproduction study are the following:

1. A generalization of E(n) Equivariant Graph Neural Networks (EGNNs), which can learn features on simplicial complexes.
2. Experiments showing that the use of higher-dimensional simplex learning improves performance compared to EGNNs and MPSNs without requiring more parameters and proving to be competitive with SOTA methods on the QM9 dataset [11], [10].

Additionally, their results suggest that incorporating geometric information serves as an effective measure against over-smoothing.

In our work, we attempt to reproduce the results of the original EMPSN paper and extend the method, rewriting parts of the author’s code to use a common suite for learning on topological domains. The suite allows us to test how a different graph lifting procedure (an operation that obtains higher-order simplices from graph data) compares to the one used in the original paper.

## 2 Theoretical background

Message passing neural networks have seen an increased popularity since their introduction [5]. In this blogpost, we will elaborate on how message passing networks are adapted to work with simplicial complexes, as proposed by [3]. We introduce the relevant definitions of message passing, simplicial complexes, equivariant message passing networks and message passing simplicial networks.

### 2.1 Message passing

<img src="images/message-passing.png" style="width: 100%;">

### 2.2 Simplicial complexes

<img src="images/simplicial-complexes.png" style="width: 100%;">

### 2.3 Equivariant Message Passing Networks

<img src="images/empns.png" style="width: 100%;">

### 2.4 Message passing simplicial networks

<img src="images/mpsns.png" style="width: 100%;">

## 3 Methodology

### 3.1 Lifted representation of the dataset

QM9 is a dataset of stable small organic molecules with geometric, energetic, and thermodynamic properties. It contains important quantum chemical properties and serves as a standard benchmark for machine learning methods or systems of identifications of the contained molecular properties. The QM9 dataset consists of molecular graphs with 19 graph-level features. The nodes of the molecular graphs are atoms embedded in a three-dimensional Euclidean space. The goal of the methodology for QM9 is graph feature prediction, and since the features are continuous values, this is a regression task. The molecular graphs present in QM9 contain no higher-order topological information. To address this limitation, we propose to lift the graph structure to facilitate the construction of different order simplices.

To do this, we *lift* the point cloud to a Vietoris-Rips complex based on a parameter $\delta$ as described in Figure 2. There is a limit on the maximum rank of the simplices; however, due to naturally occurring phenomena, it can be constrained. For this particular set of experiments, it is restricted to rank 2. Additionally, this work presents two types of communication among simplices: 1) from $r$-simplices to $r$-simplices and 2) from $(r-1)$-simplices to $r$-simplices, however, we do not use $2$-simplices to $2$-simplices, as in [3]. It is not exactly clear why, other than the increased computational cost might be too high for the added benefit.

In addition to lifting of the structure to a higher-order topological structure, we will also lift the features to embed each $r$-simplex with a certain feature vector. We experiment with three methods: 1) A sum projection of the lower boundary of the simplex; 2) a mean projection of the lower boundary simplices; and 3) a mean of the 0-simplex components composing all lower boundary simplices, as shown in the figure and as worked out in [3]. Additionally, we also used a lift to the Alpha Complex. The alpha complex is a subcomplex of the Čech complex under the condition that the radius of the Čech complex is chosen to be $\sqrt{\alpha}$, where $\alpha$ is the parameter defining the alpha complex. Identical to the simplicial complexes, Alpha complexes are constructed until the second order. The motivation for this upper limit is to contain computational demands of training EMPSN within reasonable bounds. The invariant information is included as abstract edge attribute information between a simplex boundary within a communication framework. The simplex boundary features are shown in the table below:

<img src="images/table-1.png" style="width=100%;">

In this table, the same point variables are used as in the original paper. $V(S_2)$ denotes the volume of the second order simplex, corresponding to the Volume feature in section 3.2 of the original paper.  As can be seen, some values occur twice in a simplex adjacency relation. The second time the same value occurs is because the volume of a 1d simplex is identical to the distance between its points.

