# Graph Attention Networks (GAT) - Lab Session

---
## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Part 1: Implement Single-Head GAT in NumPy](#part-1-implement-single-head-gat-in-numpy)
3. [Part 2: PyTorch Geometric GATConv](#part-2-pytorch-geometric-gatconv)
4. [Part 3: Visualize Attention Weights](#part-3-visualize-attention-weights)
5. [Part 4: Comparison](#comparison)
6. [Deliverables](#deliverables)

---

## Setup Instructions (you can jump this)

1. **Install Required Packages:**
   ```bash
   pip install numpy matplotlib networkx torch torch-geometric scikit-learn pandas seaborn

## Part 1: Implement Single-Head GAT in NumPy

Learning Objectives:
- Understand the attention mechanism formula
- Implement masked attention for graph structure
- Handle neighborhood aggregation
- Compute attention coefficients using LeakyReLU

### Exercise 1.1: Initialize GAT Layer Parameters

Task: In the GATLayerNumPy class, initialize the weight matrix W and attention vector a using Xavier initialization.
Hint: Use np.random.uniform with appropriate limits.

### Exercise 1.2: Implement LeakyReLU
Task: Complete the leaky_relu method.
Hint: Use np.where or np.maximum.

### Exercise 1.3: Compute Attention Coefficients
Task: In compute_attention_coefficients, implement the following steps:
- Compute attention logits: e_ij = a^T [Wh_i || Wh_j] 
- Apply LeakyReLU to the logits
- Mask attention to only existing edges
- Apply softmax per row (numerically stable)
Hint: Use broadcasting for efficient computation.

### Exercise 1.4: Forward Pass
Task: Complete the forward method to aggregate neighbor features using attention weights and apply an activation function.
Hint: Use ReLU or ELU for the activation.

Quiz 1.1
Question: Why do we need to mask attention coefficients using the adjacency matrix?
* To make the model faster
* To ensure nodes only attend to their neighbors
* To prevent overfitting
* To normalize the attention weights


## Part 2: PyTorch Geometric GATConv
Learning Objectives:
- Build multi-layer GAT architectures
- Use multi-head attention
- Train on real citation networks
- Evaluate model performance

**Use the Cora dataset**

### Exercise 2.1: Define GAT Architecture
Task: In the GAT class, define two GATConv layers:
- Layer 1: in_channels → hidden_channels (with heads attention heads)
- Layer 2: hidden_channels * heads → out_channels (single head, concat=False)

### Exercise 2.2: Implement Forward Pass
Task: Complete the forward method:
- Apply dropout to input
- First GAT layer + ELU activation
- Apply dropout
- Second GAT layer
- Apply log_softmax for classification

### Exercise 2.3: Training Loop
Task: Complete the train function:
- Forward pass
- Compute loss (use F.nll_loss with train_mask)
- Backward pass
- Optimizer step

Quiz 2.1
Question: Why is dropout particularly important for GAT models?
- It makes training faster
- GATs have many parameters and tend to overfit on small datasets
- It improves attention mechanism
- It reduces memory usage


## Part 3: Visualize Attention Weights
Learning Objectives:
- Extract attention weights from trained model
- Visualize attention in graph neighborhoods
- Analyze attention distribution across heads

### Exercise 3.1: Extract Attention Weights
Task: Use the trained model to extract attention weights from the first GAT layer.
Hint: Use return_attention_weights=True in the forward pass.

### Exercise 3.2: Analyze Attention Statistics
Task: Print the mean and standard deviation of attention weights for each head.
Hint: Use attention_np[:, h].mean() and attention_np[:, h].std().

### Exercise 3.3: Visualize Attention
Task: Use the provided visualize_karate_attention function to visualize attention from a specific node in the Karate Club graph.
- do similarly on Cora dataset
Hint: Call the function with node_idx=0 for the first node.

## Comparison 
Learning objectives:
- Comparison of GATv1, GATv2 and GCN

Task: Perform an experimental study to compare GCN, GAT, GATv2 on at least 2 datasets.
- Discuss and set the hyperparameters
- Report both classification and time metrics
- Examine the attention weights between GATv1 and GATv2 (for few nodes well chosen)


## Deliverables
Submit a Jupyter Notebook with:
- All Tasks completed
- Answers to all quiz questions (in comments)
- Visualizations saved as PNG files
- A short report (1-2 paragraphs) on: What you learned about attention mechanisms
Any challenges you faced and how you solved them.
- **An appendix on your use of gen AI for this lab session**:
    - Specify its usage 
    - What did it bring to you ?
    - The potential error(s)
    - The limitation(s)