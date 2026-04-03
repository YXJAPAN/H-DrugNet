# H-DrugNet

H-DrugNet is a deep learning framework designed to predict the frequency of drug side effects. By integrating a Hierarchical Representation strategy with Differentiable Pooling (DiffPool), the model effectively captures multi-level structural dependencies within drug molecules and their pharmacological associations.

---

## 🛠 Requirements

* torch==1.12.1+cu113
* torch-geometric==1.7.2
* torch-cluster==1.6.0
* torch-scatter==2.0.9
* torch-sparse==0.6.15
* rdkit==2022.9.4
* numpy==1.22.4
* pandas==1.5.1
* scipy==1.10.1
* scikit-learn==1.2.1
* networkx==2.8.8

---

##  Files

### 1. Data

This folder contains the standardized drug-side effect datasets and preprocessing matrices.

* **frequency_data.txt**
  The standardized drug side effect frequency classes used in our study.

* **drug_SMILES_750.csv**
  The SMILES representations of 750 drugs.

* **raw_frequency_750.mat**
  The original matrix of drug-adverse effect frequencies (highly sparse).

* **side_effect_label_750.mat**
  The encoded features of side effects.

* **mask_mat_750.mat**
  The mask matrix for ten-fold cross-validation in a warm-start scenario.

* **blind_mask_mat_750.mat**
  The mask matrix for ten-fold cross-validation in a cold-start scenario.

---

### 2. Experimental Scenarios

* **warm-scene_data/**
  Contains side effects and drug data specific to the **warm-start scenario**.

* **cold-scene_data/**
  Contains side effects and drug data specific to the **cold-start (blind) scenario**.

---

### 3. Code Implementation

* **Net.py**
  Defines the main architecture of the H-DrugNet model.

* **diffpool_layer.py**
  Implements the Differentiable Pooling layer:

  * Learns node clustering for hierarchical representation
  * Computes auxiliary **Link Prediction Loss** and **Entropy Loss**

* **vector.py**
  Converts drug SMILES into graph representations (nodes and edges) for GNN processing.

* **utils.py**
  Provides utility functions for:

  * Data loading
  * Evaluation metrics
  * General preprocessing

* **warm-scene.py**
  Execution script for the warm-start scenario
  *(750 drugs, 994 side effects)*

* **cold-scene.py**
  Execution script for the cold-start scenario
  *(750 drugs, 994 side effects)*

---

##  Methodology: Hierarchical Pooling

The core innovation of **H-DrugNet** lies in modeling drugs at multiple levels of granularity.

### Key Mechanisms

* **Aggregate Atoms**
  Learns an assignment matrix to group atoms into higher-level functional motifs.

* **Structural Integrity**
  Uses **Link Prediction Loss** to preserve the topological structure of the original graph after pooling.

* **Deterministic Assignment**
  Applies **Entropy Loss** to reduce uncertainty in node-to-cluster assignments.

---

##  Contact

If you have any questions or suggestions regarding the code or the research:

**Changming Yao**
China University of Geosciences (Wuhan)

📧 Email: [yaochangming@cug.edu.cn]
🆔 ORCID: 0009-0003-5928-2647
