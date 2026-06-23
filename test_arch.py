import torch
from torch_geometric.loader import DataLoader
from src.model.gnn import Tox21GNN
from src.model.loss import MaskedFocalLoss

def test():
    data_splits = torch.load('data/processed/graphs.pt')
    train_graphs = data_splits['train']
    
    NUM_NODE_FEATURES = train_graphs[0].x.shape[1]
    
    model = Tox21GNN(num_node_features=NUM_NODE_FEATURES)
    loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    
    print("Batch input x shape:", batch.x.shape)
    print("Batch input edge_attr shape:", batch.edge_attr.shape)
    
    logits = model(batch)
    print("Logits shape:", logits.shape)
    
    loss_fn = MaskedFocalLoss(alpha=torch.ones(12))
    loss = loss_fn(logits, batch.y.view(-1, 12))
    print("Loss value:", loss.item())
    
    loss.backward()
    print("Backward pass successful!")

if __name__ == "__main__":
    test()
