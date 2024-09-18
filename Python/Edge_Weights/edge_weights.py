import networkx as nx
import torch
from torch_geometric.data import Data

# **1. Create a NetworkX graph with edge weights**
G = nx.Graph()
G.add_edge(0, 1, weight=2.0)
G.add_edge(1, 2, weight=3.0)
G.add_edge(2, 0, weight=1.5)

# **2. Extract edge indices and weights**
edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
edge_attr = torch.tensor([attr['weight'] for _, _, attr in G.edges(data=True)], dtype=torch.float)

# **3. Create the PyTorch Geometric Data object**
data = Data(edge_index=edge_index, edge_weight=edge_attr) 

# You can access the edge weights like this:
print(data.edge_weight)