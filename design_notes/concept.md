Peer Normalize Kernel Based Deep Isolation Forest
- Weighted mean of K-NN peers
- soft conditioning RBF kernel weights => also, always find K-peers
- IF/DIF Foundation on normalized features
    - Input --> Random MLP  --> Representation --> IF
    - Random MLP = No training, weights fixed at random initialization
    - It acts as random feature extractors -> create diverse, non-linear projections of the input data
    - Multiples times = if one projection misses an anomaly, another catches it. MLP1 might focus amount, MLP2 focus velocity, MLP3 focus interaction...
    - Frozen or Random = No assumption on anomaly => Hence, work for any anomaly type.
    - H = leaky_relu(X @ W + b) , W = random weights, b = zero bias

- IF vs DIF
    - IF split on amount < 25000 and Velocity < 10 (limit to axis-aligned splits)
    - DIF anomalies in diagonal/curved regions

***


Con:
1. the curse of dimensionality in context k
- if there are lots of k, the space becomes empty => we won't be able to find enough neighbours to calculate a stable mean/variance for normalization.
2. Assume simple relationship
- this assume the context just shifts or scales the data
- this might not work if a "high-risk" context doesn't just shift the mean. 
- unlike dl, this can't learn a complex non-linear function of the context k
3. Need to pre-define the distance metric i.e. Kernel and Neighbor

***

Concrete Example Customer: 
- 23-year-old student in US 
- Behavior: $50,000 wire transfer, 2 transactions/month 

Step 2-3: Find 100 nearest peers (other young US students) 
Their typical: $500 transfers, 15 tx/month 

Kernel = knn / rff

Step 4: Peer stats 
μ_peer = [$500, 15] 
σ_peer = [$200, 5] 

Step 5: Normalize 
z = [($50,000 - $500) / $200, (2 - 15) / 5] 
z = [247.5, -2.6] ← Massive outlier in amount! 

Step 6: Random MLP Project (Deep part)
- h_i = MLP(z_i) -> 64-dim representation
- 6 of these == 6 vies ==> AVG(IF(h1), IF(h2),...)

Step 7: IF 
- sees z = [247.5, -2.6] → Very anomalous Without context 
- (regular IF): Global stats: μ = [$5,000, 10], σ = [$3,000, 8] z = [15, -1] ← Moderately unusual, might miss it

