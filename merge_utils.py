import json

import numpy as np
import torch


def find_similar_anchors(task_name, topk):
    with open(f"./data/{task_name}/user_anchor_candidate.json", 'r') as f:
        anchor_user_data = json.load(f)

    # Load the user embeddings.
    test_user_embs = torch.load(f"./anchor_selection/{task_name}/test_user_history_emb.pt")  # [m, h]
    anchor_user_embs = torch.load(f"./anchor_selection/{task_name}/user_history_emb.pt")     # [n, h]

    # Select the subset of anchor user embeddings.
    anchor_idxs = torch.load(f"./anchor_selection/{task_name}/anchor_user_idx.pt")
    anchor_user_ids = [anchor_user_data[anchor_idx]["user_id"] for anchor_idx in anchor_idxs]
    anchor_user_embs = anchor_user_embs[anchor_idxs, :]

    # Compute [m, n] similarity matrix.
    t_n = test_user_embs.norm(dim=1)[:, None]
    a_n = anchor_user_embs.norm(dim=1)[:, None]
    test_user_norm = test_user_embs / torch.clamp(t_n, min=1e-8)
    anchor_user_norm = anchor_user_embs / torch.clamp(a_n, min=1e-8)
    sim_mat = torch.mm(test_user_norm, anchor_user_norm.transpose(0, 1))  # [m, n]

    k = topk or anchor_user_embs.shape[0]
    topk_sims, topk_idxs = torch.topk(sim_mat, k=k, dim=1)  # [m, k]
    return np.asarray(anchor_user_ids), topk_sims.numpy(), topk_idxs.numpy()
