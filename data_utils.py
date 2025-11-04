"""Data processing utils."""

import copy

from rank_bm25 import BM25Okapi
import torch
from torch.utils.data import Dataset

from engine_finetuning import get_tokenizer
from utils import get_first_k_tokens


def process_profile(user, prompt_template, all_profile, task_name, k=1, recent_k=0):
    """Processes user profile to construct training data."""
    train_data = []
    should_format = task_name in ("movie_tagging", "news_headline", "product_rating", "scholarly_title")
    user_id = str(user["user_id"])
    user_profile = all_profile[user_id]

    user_history = user["profile"][-recent_k:]
    for idx, q in enumerate(user_history):
        for key, value in q.items():
            if isinstance(value, str):
                q[key] = get_first_k_tokens(value, 768)

        prompt = prompt_template[task_name]["OPPU_input"].format(**q)
        full_prompt = prompt_template[task_name]["OPPU_full"].format(**q)
        pid_str = f"u{user_id}-{q['id']}"
        train_data.append({"prompt": prompt, "full_prompt": full_prompt, "id": pid_str})

        if idx != 0 and should_format:
            visible_history_list = user_history[:idx]
            for p in visible_history_list:
                for key, value in p.items():
                    if isinstance(value, str):
                        p[key] = get_first_k_tokens(value, 368)

            history_list = [prompt_template[task_name]["retrieval_history"].format(**p) for p in visible_history_list]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = prompt_template[task_name]["retrieval_query"].format(**q).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)

            history_string = "".join(retrieved_history)

            prompt = f"### User History:\n{history_string}\n\n" + prompt
            full_prompt = f"### User History:\n{history_string}\n\n" + full_prompt

            prompt = f"### User Profile:\n{user_profile}\n\n" + prompt
            full_prompt = f"### User Profile:\n{user_profile}\n\n" + full_prompt

            train_data.append({"prompt": prompt, "full_prompt": full_prompt, "id": f"{pid_str}-k{k}-p"})

    return train_data


class InstructionDataset(Dataset):

    def __init__(self, data_list, tokenizer_path, max_tokens=2048):
        self.ann = data_list
        self.max_words = max_tokens
        self.tokenizer = get_tokenizer(tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        prompt = ann["prompt"]
        example = ann["full_prompt"]

        prompt = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        return example, labels, prompt


def collate_fn(batch, max_length=2048):
    examples, labels, _ = zip(*batch)
    # Trim sequences to max_length.
    trimmed_examples = [example[:max_length] for example in examples]
    trimmed_labels = [label[:max_length] for label in labels]
    
    # Determine the maximum sequence length after trimming but capped at max_length.
    max_length = min(max([len(example) for example in trimmed_examples]), max_length)

    # Pad sequences to the determined max_length.
    padded_examples = torch.stack([
        torch.cat((example, torch.zeros(max_length - len(example), dtype=torch.int64) - 1)) if len(example) < max_length else example
        for example in trimmed_examples
    ])
    padded_labels = torch.stack([
        torch.cat((label, torch.zeros(max_length - len(label), dtype=torch.int64) - 1)) if len(label) < max_length else label
        for label in trimmed_labels
    ])

    example_masks = padded_examples.ge(0)
    label_masks = padded_labels.ge(0)

    padded_examples[~example_masks] = 0
    padded_labels[~label_masks] = 0

    example_masks = example_masks.float()
    label_masks = label_masks.float()

    return padded_examples, padded_labels, example_masks
