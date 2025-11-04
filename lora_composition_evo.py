"""Evolutionary model merging for LLM personalization."""

import argparse
from collections import defaultdict
import datetime
import json
import os
from pathlib import Path
import re
import sys
sys.path.append("..")
import time

from accelerate import Accelerator
import nevergrad as ng
import numpy as np
from rank_bm25 import BM25Okapi
import torch
from tqdm import tqdm

from data_utils import process_profile
from engine_finetuning import load_model, load_generator_from_trained
from eval import evaluate_task_predictions
from merge_utils import find_similar_anchors
from utils import (
    extract_citation_title,
    extract_option,
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
    split_batch,
    get_first_k_tokens,
    name2taskid,
)

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


def get_args_parser():
    parser = argparse.ArgumentParser("LLM p13n with evolutionary merging.", add_help=False)
    parser.add_argument("--config_file", default=None, type=str, help="Config yaml file")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters.
    parser.add_argument("--llama_model_path", default=None, type=str, help="path of llama model")
    parser.add_argument("--tokenizer_path", default=None, type=str, help="path of llama model tokenizer")
    parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=3500, metavar="LENGTH", help="the maximum sequence length")
    parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (absolute lr)")
    parser.add_argument("--clip", type=float, default=0.3, help="gradient clipping")
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")

    # Dataset parameters
    parser.add_argument("--output_dir", default="./output/movie_tagging/evo-LoRA-Composition", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_ckpt", default='./output/movie_tagging/task-base_LLM/lora_ckpt.pt', help="resume lora from checkpoint")
    parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to use gradient checkpoint, recommend TRUE!!")

    parser.add_argument("--anchor_dir", default='./output/movie_tagging/Anchor_PEFT/LoRA', help="resume lora from checkpoint")
    parser.add_argument("--test_idx_dir", default='./anchor_selection/history_avg/anchor_user_idx.pt', help="resume lora from checkpoint")

    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Generation hyperparameters.
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature")
    # This is really only relevant for citation, as the training task differs from the test task.
    parser.add_argument("--train_gen_len", type=int, default=100, help="Max generation length.")
    parser.add_argument("--max_gen_len", type=int, default=100, help="top_p")
    parser.add_argument("--k_list", type=str, default="1,2,4", help="RAG k.")
    parser.add_argument('--infer', default=False, action=argparse.BooleanOptionalAction)

    # Lora composition hyperparameters.
    parser.add_argument("--topk", type=int, default=1, help="top_p")
    parser.add_argument("--recent_k", type=int, default=50, help="recent k")
    parser.add_argument("--shared_ratio", type=float, default=1, help="shared ratio")

    parser.add_argument("--opt_type", default="ngopt", type=str, help="optimizer type")
    parser.add_argument("--piece_merge", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--minimize", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_repeat", type=int, default=1, help="number of repetitions per eval")
    parser.add_argument("--opt_budget", type=int, default=100, help="optimization budget")
    parser.add_argument("--m_reduce", default="sum", type=str, help="reduction across metrics")
    parser.add_argument("--t_reduce", default="mean", type=str, help="reduction across trials")
    parser.add_argument("--train_sample", type=float, default=None, help="percentag of training data to sample")
    parser.add_argument("--n_sim", default=3, type=int, help="number of top similar anchors to use.")
    parser.add_argument("--sim_penalty", type=float, default=0.0, help="privacy penalty coefficient")
    parser.add_argument("--user_idxs", default="0-100", type=str, help="subset of test users to run on")

    return parser


def process_profile_test_data(user, batch_size, k_list):
    out_list = []
    test_question_list = []
    question_id_list = []
    retrieval_test_question_list = [[] for _ in range(len(k_list))]

    if args.task_name == "citation":
        extract_article = extract_citation_title
    elif args.task_name == "movie_tagging":
        extract_article = extract_movie
    elif args.task_name == "news_categorize":
        extract_article = extract_news_cat
    elif args.task_name == "news_headline":
        extract_article = extract_news_headline
    elif args.task_name == "product_rating":
        extract_article = extract_product_review
    elif args.task_name == "scholarly_title":
        extract_article = extract_scholarly_title
    elif args.task_name == "tweet_paraphrase":
        extract_article = extract_tweet_paraphrasing

    with open("./prompt/prompt.json", "r") as f:
        prompt_template = json.load(f)

    user_profile = all_profile[str(user["user_id"])]

    for q in user["query"]:
        if args.task_name == "citation":
            test_question = q["input"]
            test_article = extract_article(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
        else:
            test_question = q["input"]
            test_article = extract_article(test_question)
            test_prompt =  prompt_template[args.task_name]["prompt"].format(test_article)

        test_prompt = f"### User Profile:\n{user_profile}\n\n" + test_prompt

        test_question_list.append(test_prompt)
        question_id_list.append(q["id"])

    visible_history_list = user["profile"]
    for p in visible_history_list:
        for key, value in p.items():
            if isinstance(value, str):
                p[key] = get_first_k_tokens(value, 368)

    history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in visible_history_list]

    tokenized_corpus = [doc.split(" ") for doc in history_list]
    bm25 = BM25Okapi(tokenized_corpus)

    for idx, k in enumerate(k_list):
        for q in user["query"]:
            if args.task_name == "citation":
                test_question = q["input"]
                test_article = extract_article(test_question)
                option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
                test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
            else:
                test_question = q["input"]
                test_article = extract_article(test_question)
                test_prompt =  prompt_template[args.task_name]["prompt"].format(test_article)

            tokenized_query = prompt_template[args.task_name]["retrieval_query_wokey"].format(test_article).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)
        
            history_string = "".join(retrieved_history)

            test_prompt = f"### User History:\n{history_string}\n\n" + test_prompt
            test_prompt = f"### User Profile:\n{user_profile}\n\n" + test_prompt

            retrieval_test_question_list[idx].append(test_prompt)

    test_batch_list = split_batch(test_question_list, batch_size)
    out_list.append(test_batch_list)

    for i, k in enumerate(k_list):
        out_list.append(split_batch(retrieval_test_question_list[i], batch_size))

    all_test_question_list = [test_question_list] + retrieval_test_question_list
    return out_list, question_id_list, all_test_question_list


def process_test_data(user, batch_size, k_list):
    out_list = []
    test_question_list = [] 
    question_id_list = []
    retrieval_test_question_list = [[] for _ in range(len(k_list))]

    if args.task_name == "citation":
        extract_article = extract_citation_title
    elif args.task_name == "movie_tagging":
        extract_article = extract_movie
    elif args.task_name == "news_categorize":
        extract_article = extract_news_cat
    elif args.task_name == "news_headline":
        extract_article = extract_news_headline
    elif args.task_name == "product_rating":
        extract_article = extract_product_review
    elif args.task_name == "scholarly_title":
        extract_article = extract_scholarly_title
    elif args.task_name == "tweet_paraphrase":
        extract_article = extract_tweet_paraphrasing

    with open("./prompt/prompt.json", "r") as f:
        prompt_template = json.load(f)

    for q in user["query"]:
        if args.task_name == "citation":
            test_question = q["input"]
            test_article = extract_article(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
        else:
            test_question = q["input"]
            test_article = extract_article(test_question)
            test_prompt =  prompt_template[args.task_name]["prompt"].format(test_article)

        test_question_list.append(test_prompt)
        question_id_list.append(q["id"])

    visible_history_list = user["profile"]
    for p in visible_history_list:
        for key, value in p.items():
            if isinstance(value, str):
                p[key] = get_first_k_tokens(value, 368)

    history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in visible_history_list]

    tokenized_corpus = [doc.split(" ") for doc in history_list]
    bm25 = BM25Okapi(tokenized_corpus)

    for idx, k in enumerate(k_list):
        for q in user["query"]:
            if args.task_name == "citation":
                test_question = q["input"]
                test_article = extract_article(test_question)
                option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
                test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
            else:
                test_question = q["input"]
                test_article = extract_article(test_question)
                test_prompt =  prompt_template[args.task_name]["prompt"].format(test_article)

            tokenized_query = prompt_template[args.task_name]["retrieval_query_wokey"].format(test_article).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)
        
            history_string = "".join(retrieved_history)

            test_prompt = f"### User History:\n{history_string}\n\n" + test_prompt

            retrieval_test_question_list[idx].append(test_prompt)

    test_batch_list = split_batch(test_question_list, batch_size)
    out_list.append(test_batch_list)

    for i, k in enumerate(k_list):
        out_list.append(split_batch(retrieval_test_question_list[i], batch_size))

    all_test_question_list = [test_question_list] + retrieval_test_question_list

    return out_list, question_id_list, all_test_question_list


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


def l1_reg(weights, scale=0.05):
    sum_of_abs = sum([abs(x) for x in weights]) / len(weights)
    return scale * sum_of_abs


def reduce(type_str, values):
    values = list(values.values()) if isinstance(values, dict) else list(values)
    if type_str == "sum": return np.sum(values)
    if type_str == "mean": return np.mean(values)
    if type_str == "max": return np.max(values)
    if type_str == "min": return np.min(values)
    return values[0]


def scores(args, eval_result, lora_eval_results=None):
    user_score = reduce(args.m_reduce, eval_result)
    lora_score = 0.0
    if lora_eval_results is not None:
        lora_eval_scores = [
            reduce(args.m_reduce, lora_eval_result)
            for lora_eval_result in lora_eval_results
        ]
        lora_score = np.mean(lora_eval_scores)

    # Calculate the final score.
    multiplier = 1.0 if args.minimize else -1.0
    loss = (user_score - args.sim_penalty * lora_score) * multiplier  # NG expects losses.
    return loss, user_score, lora_score


def trial(args, model, data_list, golds_dict, lora_preds_dicts):
    generator = load_generator_from_trained(model, args.tokenizer_path)
    max_gen_len = args.train_gen_len if args.task_name == "citation" else args.max_gen_len
    preds_dict = predict(generator, data_list, args.task_name,
                         args.batch_size, max_gen_len, args.temperature, args.top_p)

    # Evaluate how well the model performs on the target user data.
    eval_result = evaluate_task_predictions(golds_dict, preds_dict)

    # Evaluate how well the model's predictions match those of anchors.
    lora_eval_results = [
        evaluate_task_predictions(lora_preds_dict, preds_dict)
        for lora_preds_dict in lora_preds_dicts
    ]

    loss, user_score, lora_score = scores(args, eval_result, lora_eval_results)
    eval_result["debug"] = {
        "loss": loss,
        "user_score": user_score,
        "lora_score": lora_score,
        "lora_eval_results": lora_eval_results,
    }
    return loss, eval_result


def get_loss(args, model, loras, weights, data_list, golds_dict, lora_preds_dicts):
    losses = []
    raw_eval_results = []
    for _ in range(args.num_repeat):
        model.get_weighted_lora(lora_corpus=loras, weights=weights)
        loss, eval_result = trial(args, model, data_list, golds_dict, lora_preds_dicts)
        raw_eval_results.append(eval_result)
        model.reset_lora_parameters()
        losses.append(loss)

    return reduce(args.t_reduce, losses), raw_eval_results


def merge_user_loras(args, loras, model, data_list, golds_dict, lora_preds_dicts):
    layer_idxs = sorted(list(set(int(re.search("layers\.(\d+)\.", n).groups()[0]) for n in loras[0].keys())))
    num_loras, num_layers = len(loras), len(layer_idxs)
    num_ws = num_loras * num_layers if args.piece_merge else num_loras
    shape = (num_loras, num_layers) if args.piece_merge else (num_loras,)

    # Set up the optimizer and limits.
    instrum = ng.p.Array(
        init=[0.0] * num_ws,
        upper=[1.5] * num_ws,
        lower=[-1.5] * num_ws,
    )
    optimizer = None
    if args.opt_type == "ngopt":
        optimizer = ng.optimizers.NGOpt(instrum, args.opt_budget)
    elif args.opt_type == "cma":
        optimizer = ng.optimizers.CMA(instrum, args.opt_budget)
    elif args.opt_type == "tbpsa":
        optimizer = ng.optimizers.TBPSA(instrum, args.opt_budget)
    else:
        raise ValueError(f"Unsupported optimizer type: {args.opt_type}")

    # Run optimization.
    best_loss = float("inf")
    best_raw_eval_results = None
    opt_results = []
    for idx in tqdm(range(optimizer.budget), total=optimizer.budget):
        x = optimizer.ask()
        weights = np.array(x.args).reshape(shape)
        loss, raw_eval_results = get_loss(args, model, loras, weights, data_list, golds_dict, lora_preds_dicts)
        if best_loss > loss:
            best_loss = loss
            best_raw_eval_results = raw_eval_results
        opt_results.append(raw_eval_results)
        optimizer.tell(x, loss)
        if args.piece_merge:
            print(f"({idx}) loss: {round(loss, 4)}")
        else:
            print(f"({idx}) weights: {[round(w, 4) for w in weights]}, loss: {round(loss, 4)}")

    # Save opt results and set weights.
    recommend = optimizer.provide_recommendation()
    weights = np.array(recommend.value).reshape(shape)
    model.get_weighted_lora(lora_corpus=loras, weights=weights)
    best_eval_result = best_raw_eval_results[-1]  # Just take the last one.
    best_eval_result["debug"]["recommend"] = list(weights)
    print(f"recommend: {weights}, eval result: {best_eval_result}")

    return opt_results, best_loss, best_eval_result


def base_eval(args, model, loras, data_list, golds_dict):
    preds_dicts = []
    for lora in [None] + loras:
        if lora is not None:
            model.get_weighted_lora(lora_corpus=[lora], weights=[1.0])
        generator = load_generator_from_trained(model, args.tokenizer_path)
        max_gen_len = args.train_gen_len if args.task_name == "citation" else args.max_gen_len
        preds_dict = predict(generator, data_list, args.task_name,
                             args.batch_size, max_gen_len, args.temperature, args.top_p)
        preds_dicts.append(preds_dict)
        model.reset_lora_parameters()

    base_preds_dict, lora_preds_dicts = preds_dicts[0], preds_dicts[1:]
    base_eval_result = evaluate_task_predictions(golds_dict, base_preds_dict)
    base_lora_eval_results = [
        evaluate_task_predictions(lora_preds_dict, base_preds_dict)
        for lora_preds_dict in lora_preds_dicts
    ]
    base_loss, base_user_score, base_lora_score = scores(args, base_eval_result, base_lora_eval_results)
    base_eval_result["debug"] = {
        "loss": base_loss,
        "user_score": base_user_score,
        "lora_score": base_lora_score,
        "lora_eval_results": base_lora_eval_results,
    }
    return base_loss, base_eval_result, lora_preds_dicts


def main(args):
    # Fix the seed for reproducibility.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Scratch directory for processes.
    tmp_dir = os.path.join(args.output_dir, "tmp")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write("{}".format(args).replace(", ", ",\n") + "\n")

    # Find top-k similar anchor users.
    anchor_user_ids, topk_sims, topk_idxs = find_similar_anchors(args.task_name, topk=args.n_sim)
    del topk_sims  # Inspect for debugging purposes.

    accelerator = Accelerator()  # For batch inference.

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
    model = accelerator.prepare(model)

    # Load anchor user LoRAs *in the right order*.
    all_loras = []
    for anchor_user_id in anchor_user_ids:
        anchor_dir = os.path.join(args.anchor_dir, f"user_{anchor_user_id}")
        assert os.path.exists(anchor_dir), f"Anchor dir not found: {anchor_dir}"
        lora = torch.load(os.path.join(anchor_dir, "lora_ckpt.pt"), map_location="cpu")
        all_loras.append(lora)

    with open(args.test_data_path, "r") as f:
        test_users = json.load(f)
    user_idxs = args.user_idxs.split("-")
    s_idx, e_idx = int(user_idxs[0]), int(user_idxs[1])
    test_users = test_users[s_idx:e_idx]

    train_golds = dict()
    train_eval_results = dict()
    opt_eval_results = dict()
    pred_all = defaultdict(dict)
    pred_profile_all = defaultdict(dict)
    chosen_anchors = defaultdict(list)

    # Test user indices should be computed deterministically prior to split.
    pidx = accelerator.process_index
    test_user_with_idx = [(idx, user) for idx, user in enumerate(test_users)]
    with accelerator.split_between_processes(test_user_with_idx) as sub_test_users:
        for idx, user in tqdm(
            sub_test_users,
            desc=f"({pidx}) Merging",
            total=len(sub_test_users),
        ):
            model.reset_lora_parameters()  # IIUC, this isn't really needed as we override params.

            user_id = str(user["user_id"])
            # Select a subset of anchors based on the pre-computed similarities.
            loras = []
            for chosen_idx in topk_idxs[idx]:
                loras.append(all_loras[chosen_idx])
                chosen_anchors[f"{user_id}-{idx}"].append(str(anchor_user_ids[chosen_idx]))

            data_list = process_profile(user, prompt_template, all_profile, args.task_name, recent_k=args.recent_k)
            if args.train_sample:
                num_samples = int(args.train_sample)
                if args.train_sample <= 1.0:
                    num_samples = int(len(data_list) * args.train_sample)
                data_list = np.random.choice(data_list, size=num_samples, replace=False)

            # Evaluate the base model prior to merging.
            # A hack to use rouge for train eval in case of citation.
            task_id = "LaMP" if args.task_name in ("citation") else name2taskid[args.task_name]
            golds_dict = {"task": task_id, "golds": []}
            for datum in data_list:
                prompt, full_prompt = datum["prompt"], datum["full_prompt"]
                golds_dict["golds"].append(
                    {
                        "id": datum["id"],
                        "output": full_prompt.replace(prompt, "").strip(),
                    }
                )
            train_golds[user_id] = golds_dict

            base_loss, base_eval_result, lora_preds_dicts = base_eval(args, model, loras, data_list, golds_dict)

            # Merge and run a final evaluation.
            start_time = time.time()
            opt_results, loss, eval_result = merge_user_loras(args, loras, model, data_list, golds_dict, lora_preds_dicts)
            total_time = time.time() - start_time
            opt_eval_results[user_id] = opt_results
            train_eval_results[user_id] = {"base": base_eval_result, "final": eval_result}
            print(f"Eval results: {train_eval_results[user_id]}")
            # Revert to the base model in case of no improvements.
            if base_loss < loss:
                model.reset_lora_parameters()

            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Merging time {}".format(total_time_str))

            # Inference stage.
            # TODO(kykim): Clean up.
            generator = load_generator_from_trained(model, args.tokenizer_path)
            test_batch_list, test_id_list, test_question_list = process_test_data(user, batch_size=args.batch_size, k_list=args.k_list)
            for idx, setting in enumerate(test_batch_list):
                all_results = []
                for batch in setting:
                    results = generator.generate(batch, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
                    all_results += results
                pred_user = []
                for i in range(len(all_results)):
                    output = all_results[i].replace(test_question_list[idx][i], "").strip()
                    pred_user.append({"id": test_id_list[i], "output": output})
                k_idx = 0 if idx == 0 else args.k_list[idx-1]
                pred_all[k_idx][user_id] = pred_user

            test_batch_list, test_id_list, test_question_list = process_profile_test_data(user, batch_size=args.batch_size, k_list=args.k_list)
            for idx, setting in enumerate(test_batch_list):
                all_results = []
                for batch in setting:
                    results = generator.generate(batch, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
                    all_results += results
                pred_profile_user = []
                for i in range(len(all_results)):
                    output = all_results[i].replace(test_question_list[idx][i], "").strip()
                    pred_profile_user.append({"id": test_id_list[i], "output": output})
                k_idx = 0 if idx == 0 else args.k_list[idx-1]
                pred_profile_all[k_idx][user_id] = pred_profile_user

        # Write out per-process results.
        for fname, data_dict in zip(
            ["train-golds.json", "train-evals.json", "opt-evals.json", "pred.json", "pred-profile.json", "chosen-anchors.json"],
            [train_golds, train_eval_results, opt_eval_results, pred_all, pred_profile_all, chosen_anchors],
        ):
            with open(os.path.join(tmp_dir, f"{pidx}-{fname}"), "w") as f:
                json.dump(data_dict, f, indent=4)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.test_data_path = f"./data/{args.task_name}/test_100/user_test_100.json"
    args.test_gold_data_path = f"./data/{args.task_name}/test_100/user_test_100_label.json"
    args.k_list = [int(k) for k in args.k_list.split(",")]
    print(args)
    with open(f"./data/{args.task_name}/profile-id2text.json", "r") as f:
        all_profile = json.load(f)
    with open(f"./prompt/prompt.json", "r") as f:
        prompt_template = json.load(f)
    main(args)
