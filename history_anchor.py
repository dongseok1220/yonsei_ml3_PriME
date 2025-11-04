# Script to compute user embeddings based on the history data.
#
# Usage:
# $ python3 history_anchor.py \
#     --task_name movie_tagging --candidate_path ./data/movie_tagging/user_anchor_candidate.json \
#     --prompt_path ./prompt/prompt.json --embedding_filename user_history_emb.pt 
# $ python3 history_anchor.py \
#     --task_name movie_tagging --candidate_path ./data/movie_tagging/test_100/user_test_100.json \
#     --prompt_path ./prompt/prompt.json --embedding_filename test_user_history_emb.pt --k 0

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
from transformers import DebertaV2Model, DebertaV2Tokenizer

from utils import get_first_k_tokens, split_batch


def get_args_parser():
    parser = argparse.ArgumentParser("Anchor selection based on user history", add_help=False)
    parser.add_argument("--candidate_path", default=None, type=str, help="path to candidate user json file")
    parser.add_argument("--prompt_path", default="./prompt/prompt.json", type=str, help="path to prompt file")
    parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")
    parser.add_argument("--embedding_filename", default="user_history_emb.pt", type=str, help="name of the output embedding file.")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--k", default=50, type=int, help="number of selected anchor user")
    return parser


def compute_user_embeddings(args, anchor_candidate, prompt_template):
    # Load the DeBERTa-v3-base tokenizer and model.
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-large").cuda()

    # Compute embeddings based on user history data.
    all_user_emb = []
    for user in tqdm(anchor_candidate):
        history_embeddings_list = []
        visible_history_list = user["profile"]
        for p in visible_history_list:
            for key, value in p.items():
                if isinstance(value, str):
                    p[key] = get_first_k_tokens(value, 368)

        user_nl_history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in visible_history_list]
        user_nl_history_list_batched = split_batch(user_nl_history_list, args.batch_size)
        for batch in user_nl_history_list_batched:
            with torch.no_grad():
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
                outputs = model(**inputs)

                # Compute the attention mask and the last hidden state.
                attention_mask = inputs["attention_mask"]       # [b, l]
                last_hidden_states = outputs.last_hidden_state  # [b, l, h]

                # Multiply last hidden states by attention mask, then sum and divide by the number of tokens.
                attention_mask = attention_mask.unsqueeze(-1)               # [b, l, 1]
                masked_hidden_states = last_hidden_states * attention_mask  # [b, l, h]
                summed = torch.sum(masked_hidden_states, 1)                 # [b, h]
                count = torch.clamp(attention_mask.sum(1), min=1e-9)        # [b, 1]
                mean_pooled = summed / count

            history_embeddings_list.append(mean_pooled.cpu())

        history_embedding_concat = torch.cat(history_embeddings_list, dim=0).cpu().mean(dim=0, keepdim=True)
        all_user_emb.append(history_embedding_concat)

    all_user_emb = torch.cat(all_user_emb, dim=0)
    print(f"Size of user embeddings: {all_user_emb.shape}")

    dirname = f"./anchor_selection/{args.task_name}"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    torch.save(all_user_emb, os.path.join(dirname, f"{args.embedding_filename}"))
    return all_user_emb.numpy()


def select_anchors(args, emb, anchor_candidate):
    k = args.k
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=3000, n_init=10).fit(emb)
    labels = kmeans.labels_

    selected_indices = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        max_len = 0
        for idx in cluster_indices:
            if len(anchor_candidate[idx]["profile"]) > max_len:
                max_len = len(anchor_candidate[idx]["profile"])
                selected_index = idx

        # Consider an anchor if more than 10 history items.
        if max_len > 10:
            selected_indices.append(selected_index)

    print(f"Number of anchor users: {len(selected_indices)}")
    torch.save(selected_indices, f"./anchor_selection/{args.task_name}/anchor_user_idx.pt")


def main(args):
    with open(args.candidate_path, "r") as f:
        anchor_candidate = json.load(f)
    with open(args.prompt_path, "r") as f:
        prompt_template = json.load(f)

    user_embds = compute_user_embeddings(args, anchor_candidate, prompt_template)
    # Select anchor users in case of a non-negative k.
    if args.k > 0: select_anchors(args, user_embds, anchor_candidate)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
