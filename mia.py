"""Loss-based membership inference attack analysis."""

import argparse
import copy
from functools import partial
import glob
import json
import os
import sys
sys.path.append("..")

import numpy as np
import torch
from tqdm import tqdm

from data_utils import collate_fn, InstructionDataset, process_profile
from engine_finetuning import load_model
from llama import Tokenizer
from merge_utils import find_similar_anchors
from utils import name2taskid, split_batch


def get_args_parser():
    parser = argparse.ArgumentParser("MIA analysis.", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")

    # Model parameters.
    parser.add_argument("--llama_model_path", default=None, type=str, help="path of llama model")
    parser.add_argument("--tokenizer_path", default=None, type=str, help="path of llama model tokenizer")
    parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=3500, metavar="LENGTH", help="the maximum sequence length")
    parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

    # Dataset parameters
    parser.add_argument("--exp_dir", default="./output/movie_tagging/LoRA-Composition", help="path where to save, empty for no saving")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_ckpt", default='./output/movie_tagging/task-base_LLM/lora_ckpt.pt', help="resume lora from checkpoint")
    parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to use gradient checkpoint, recommend TRUE!!")

    parser.add_argument("--anchor_dir", default='./output/movie_tagging/Anchor_PEFT/LoRA', help="resume lora from checkpoint")
    parser.add_argument("--gate_dir", default='./output/movie_tagging/Anchor_PEFT/gate', help="resume lora from checkpoint")
    parser.add_argument("--test_idx_dir", default='./anchor_selection/history_avg/anchor_user_idx.pt', help="resume lora from checkpoint")

    parser.add_argument("--num_workers", default=10, type=int)

    # Generation hyperparameters.
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature")
    parser.add_argument("--max_gen_len", type=int, default=100, help="Max generation length.")
    parser.add_argument("--k_list", type=str, default="1,2,4", help="RAG k.")

    # Lora composition hyperparameters.
    parser.add_argument("--topk", type=int, default=1, help="top_k")
    parser.add_argument("--recent_k", type=int, default=50, help="No. of recent profile items to use for training.")
    parser.add_argument("--agg_temperature", type=float, default=1, help="temperature")
    parser.add_argument('--sample', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sample_topk", type=int, default=10, help="topk")
    parser.add_argument("--sample_temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--sample_top_p", type=float, default=None, help="Sampling topd p")
    parser.add_argument("--shared_ratio", type=float, default=1, help="shared ratio")

    parser.add_argument("--n_sim", default=None, type=int, help="no. of top similar anchors to use.")
    parser.add_argument('--is_prime', default=False, action=argparse.BooleanOptionalAction)

    return parser


def get_all_history_id(data, tokenizer_path, max_length):
    tokenizer = Tokenizer(model_path=tokenizer_path + "/tokenizer.model")

    example_all = []
    label_all = []

    for ann in data:
        prompt = ann['prompt']
        example = ann['full_prompt']

        prompt = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        example_all.append(example)

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        label_all.append(labels)

    trimmed_examples = [example[:max_length] for example in example_all]
    trimmed_labels = [label[:max_length] for label in label_all]

    # Determine the maximum sequence length after trimming but capped at max_length
    max_length = min(max([len(example) for example in trimmed_examples]), max_length)

    # Pad sequences to the determined max_length
    padded_examples = torch.stack([torch.cat((example, torch.zeros(max_length - len(example), dtype=torch.int64) - 1)) if len(example) < max_length else example for example in trimmed_examples])
    padded_labels = torch.stack([torch.cat((label, torch.zeros(max_length - len(label), dtype=torch.int64) - 1)) if len(label) < max_length else label for label in trimmed_labels])

    example_masks = padded_examples.ge(0)
    label_masks = padded_labels.ge(0)

    padded_examples[~example_masks] = 0
    padded_labels[~label_masks] = 0

    return padded_examples, padded_labels


# TODO(kykim): Refactor.
def predict(generator, data_list, task_name,
            batch_size, max_gen_len, temperature, top_p):
    """Computes model predictions on a given data.

    The input data is expected to be a list of dicts, with each dict containing
    the prompt, the full prompt (including the label), and the example id.
    """
    prompts = [datum["prompt"] for datum in data_list]
    train_batch_list = split_batch(prompts, batch_size)

    all_preds = []
    for batch in train_batch_list:
        results = generator.generate(batch, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        all_preds += results

    # A hack to use rouge for train eval in case of citation.
    task_id = "LaMP" if task_name in ("citation") else name2taskid[task_name]
    preds_dict = {"task": task_id, "golds": []}
    for datum, pred in zip(data_list, all_preds):
        output = pred.replace(datum["prompt"], "").strip()
        preds_dict["golds"].append({"id": datum["id"], "output": output})

    return preds_dict


def compute_loss(model, data_loader):
    """Returns language modeling losses for given samples."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    losses = []
    for _, (examples, labels, _) in enumerate(data_loader):
        with torch.no_grad():
            start_pos = 0
            _bsz, seqlen = examples.shape
            h = model.tok_embeddings(examples)
            model.freqs_cis = model.freqs_cis.to(h.device)
            freqs_cis = model.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full(
                    (seqlen, seqlen), float("-inf"), device=examples.device, dtype=torch.float32
                )

                mask = torch.triu(mask, diagonal=1)

                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                mask = torch.hstack([
                    torch.zeros((seqlen, start_pos), device=examples.device),
                    mask
                ]).type_as(h)

            for layer in model.layers:
                h = layer(h, start_pos, freqs_cis, mask)
            h = model.norm(h)
            output = model.output(h).float()

            output = output[:, :-1, :].reshape(-1, model.vocab_size)
            labels = labels[:, 1:].flatten()

            c_loss = criterion(output, labels)
            c_loss = torch.sum(c_loss.reshape(_bsz, -1), dim=-1, keepdim=False)
            c_loss_value = c_loss.cpu().numpy().tolist()
            losses.extend(c_loss_value)
    return losses


def main(args):
    # Fix the seed for reproducibility.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load the json containing chosen anchor info.
    chosen_anchor_dict = {}
    chosen_anchor_jsons = glob.glob(os.path.join(args.exp_dir, "tmp", "*-chosen-anchors.json"))
    for chosen_anchor_json in chosen_anchor_jsons:
        with open(chosen_anchor_json, "r" ) as f:
            loaded_dict = json.load(f)
        for k, v in loaded_dict.items():  # k: {user_id}-{idx}
            chosen_anchor_dict[k] = v

    # Load the json containing train eval info for PriME.
    weights_dict = {}
    train_eval_jsons = glob.glob(os.path.join(args.exp_dir, "tmp", "*-train-evals.json"))
    for train_eval_json in train_eval_jsons:
        with open(train_eval_json, "r" ) as f:
            loaded_dict = json.load(f)
        for k, v in loaded_dict.items():  # k: {user_id}
            if "final" in v and "debug" in v["final"] and "recommend" in v["final"]["debug"]:
                weights_dict[k] = v["final"]["debug"]["recommend"]
    args.is_prime = bool(weights_dict)

    # Find top-k similar anchor users.
    anchor_user_ids, topk_sims, topk_idxs = find_similar_anchors(args.task_name, topk=None)
    del topk_sims  # Inspect for debugging purposes.

    # Load the model.
    model = load_model(
        ckpt_dir=args.llama_model_path,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        lora_path=args.lora_ckpt,
        w_lora=args.w_lora,
        grad_ckpt=args.grad_ckpt,
    )
    model.merge_lora_parameters()
    model.set_all_frozen()

    # Load anchor user LoRAs *in the right order*.
    all_loras = []
    anchor_data_dict = {}
    for anchor_user_id in anchor_user_ids:
        anchor_user_id = str(anchor_user_id)
        anchor_dir = os.path.join(args.anchor_dir, f"user_{anchor_user_id}")
        assert os.path.exists(anchor_dir), f"Anchor dir not found: {anchor_dir}"
        lora = torch.load(os.path.join(anchor_dir, "lora_ckpt.pt"), map_location="cpu")
        all_loras.append(lora)
        anchor_user = anchor_users[anchor_user_id]
        anchor_data_list = process_profile(anchor_user, prompt_template, all_profile, args.task_name, recent_k=args.recent_k)
        anchor_data_dict[anchor_user_id] = anchor_data_list

    mia_out_dict = {}
    test_user_with_idx = [(idx, user) for idx, user in enumerate(test_users)]
    for idx, user in tqdm(test_user_with_idx, desc=f"Mia", total=len(test_user_with_idx)):
        model.reset_lora_parameters()  # IIUC, this isn't really needed as we override params.

        user_id = str(user["user_id"])
        user_id_idx = f"{user_id}-{idx}"
        user_out_dir = os.path.join(args.exp_dir, "loras", f"user_{user_id}")

        topk = args.n_sim or len(topk_idxs[idx])

        loras, chosen_anchor_ids = [], []
        for chosen_idx in topk_idxs[idx][:topk]:
            anchor_user_id = str(anchor_user_ids[chosen_idx])
            chosen_anchor_ids.append(anchor_user_id)
            loras.append(all_loras[chosen_idx])
        non_chosen_anchor_ids = [str(anchor_user_ids[aidx]) for aidx in topk_idxs[idx][topk:]]

        # Load in the LoRA for the test user.
        user_lora_ckpt = os.path.join(user_out_dir, "lora_ckpt.pt")
        if os.path.exists(user_lora_ckpt):
            lora = torch.load(user_lora_ckpt, map_location="cpu")
            model.get_weighted_lora(lora_corpus=[lora], weights=[1.0])
        else:
            if args.is_prime:
                model.get_weighted_lora(lora_corpus=loras, weights=weights_dict[user_id])
            else:
                lora_path_list = [os.path.join(args.anchor_dir, f"user_{uid}", "lora_ckpt.pt") for uid in chosen_anchor_ids]
                gate_path_list = [os.path.join(args.gate_dir, f"user_{uid}", "gate_ckpt.pt") for uid in chosen_anchor_ids]

                # Evaluate the base model prior to merging.
                data_list = process_profile(user, prompt_template, all_profile, args.task_name, recent_k=args.recent_k)

                # Merge and run a final evaluation.
                input_ids, labels = get_all_history_id(data_list, args.tokenizer_path, args.max_seq_len)
                model.get_new_lora(
                    lora_path_list=lora_path_list,
                    gate_path_list=gate_path_list,
                    input_ids=input_ids,
                    labels=labels,
                    batch_size=args.batch_size,
                    topk=args.topk,
                    epoch=args.epochs,
                    temperature=args.agg_temperature,
                    sample=args.sample,
                    sample_topk=args.sample_topk,
                    sample_temperature=args.sample_temperature,
                    sample_top_p=args.sample_top_p,
                    shared_ratio=args.shared_ratio
                )

        member_samples = [anchor_data_dict[aid] for aid in chosen_anchor_ids]
        non_member_samples = [anchor_data_dict[aid] for aid in non_chosen_anchor_ids]
        member_samples_all = [e for l in member_samples for e in l]
        non_member_samples_all = [e for l in non_member_samples for e in l]

        member_losses, non_member_losses = [], []
        for data_list, losses in zip(
            [member_samples_all, non_member_samples_all],
            [member_losses, non_member_losses],
        ):
            dataset_train = InstructionDataset(
                data_list=data_list, tokenizer_path=args.tokenizer_path, max_tokens=args.max_seq_len
            )
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                shuffle=True,
                batch_size=args.batch_size,
                drop_last=False,
                generator=torch.Generator(device="cuda"),
                collate_fn=partial(collate_fn, max_length=args.max_seq_len),
            )
            lm_losses = compute_loss(model, data_loader_train)
            losses.extend(lm_losses)

        mia_out_dict[user_id_idx] = {
            "member_ids": chosen_anchor_ids,
            "non_member_ids": non_chosen_anchor_ids,
            "member_num_samples": [len(l) for l in member_samples],
            "non_member_num_samples": [len(l) for l in non_member_samples],
            "member_loss": member_losses,
            "non_member_loss": non_member_losses,
        }

    with open(os.path.join(args.exp_dir, "mia_results.json"), "w") as f:
        json.dump(mia_out_dict, f, indent=4)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.test_data_path = f"./data/{args.task_name}/test_100/user_test_100.json"
    args.test_gold_data_path = f"./data/{args.task_name}/test_100/user_test_100_label.json"
    args.k_list = [int(k) for k in args.k_list.split(",")]
    print(args)
    with open(f'./data/{args.task_name}/profile-id2text.json', 'r') as f:
        all_profile = json.load(f)
    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)
    with open(args.test_data_path, "r") as f:
        test_users = json.load(f)
    anchor_users = {}
    with open(f'./data/{args.task_name}/user_anchor_candidate.json', 'r') as f:
        anchor_candidates = json.load(f)
        for anchor in anchor_candidates:
            anchor_users[str(anchor["user_id"])] = anchor
    main(args)
