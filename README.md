# Optimizing Random Vectors for Orthogonality 

**Step 1: Initialize and Normalize Vectors**

1) A random tensor (big_matrix) of size [250 x 10] is created, where 250 is the number of vectors and 10 is their dimension.

2) Each vector is normalized to have a unit length, ensuring they lie on the surface of a hypersphere.
```python

big_matrix = torch.rand(num_vectors, vector_len)
big_matrix = big_matrix / big_matrix.norm(2, dim=1, keepdim=True)
big_matrix.requires_grad_(True) 
```
**Step 2: Visualize Initial Vectors**

1) The first two components of each vector are plotted in 2D using Matplotlib's quiver function.
   
2) This shows the random distribution of vectors before optimization, where significant overlap (superposition) is visible.
 ```python
   plt.figure(figsize=(8, 8))  
for i in range(num_vectors):  
    plt.quiver(0, 0, big_matrix[i, 0].item(), big_matrix[i, 1].item(), angles='xy', scale_units='xy', scale=1, color='b', alpha=0.5)  
plt.xlim(-1, 1)  
plt.ylim(-1, 1)  
plt.title("Initial Random Vectors (Superposition)")  
plt.grid(True)  
plt.show()
```
**Step 3: Optimization for Orthogonality**

**Objective:** Minimize the overlap (dot product) between vectors while maintaining their unit length.

The optimization is driven by:
1) Penalizing non-zero off-diagonal values in the dot-product matrix.
2) Keeping diagonal values close to 1, as they represent the self-dot product of unit vectors.
 
```
for step_num in range(num_steps):  
    optimizer.zero_grad()  
    dot_products = big_matrix @ big_matrix.T  
    diff = dot_products - big_id  
    loss = (diff.abs() - dot_diff_cutoff).relu().sum()  
    loss += num_vectors * diff.diag().pow(2).sum()  
    loss.backward()  
    optimizer.step()  
    losses.append(loss.item())  
```
**Step 4: Visualize Optimized Vectors**
1) After optimization, the first two components of the vectors are plotted again.
2) The optimized vectors exhibit reduced overlap, showing a more orthogonal configuration.
```
   plt.figure(figsize=(8, 8))  
for i in range(num_vectors):  
    plt.quiver(0, 0, big_matrix[i, 0].item(), big_matrix[i, 1].item(), angles='xy', scale_units='xy', scale=1, color='r', alpha=0.5)  
plt.xlim(-1, 1)  
plt.ylim(-1, 1)  
plt.title("Vectors After Optimization (Reduced Superposition)")  
plt.grid(True)  
plt.show()  
```
