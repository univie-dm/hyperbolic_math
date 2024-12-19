import torch
from src.manifolds import PoincareBall
from src.manifolds.embedding import Embedding


if __name__ == "__main__":
    manifold = PoincareBall()
    for m in ["dist2hyperplane", "direct_matvec_mul", "indirect_matvec_mul", "hnn_matvec_mul"]:
        model = Embedding(256, 4, manifold, c=1.0, forward_method=m)
        test = torch.randn(32, 256)
        out = model(test)
        assert out.shape == (32, 4)