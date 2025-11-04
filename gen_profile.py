"""Script for generating user profiles."""

import argparse
import json
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import split_batch, get_first_k_tokens


def get_args_parser():
    parser = argparse.ArgumentParser(description="Profile generation")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cut_off", type=int, default=2048)
    parser.add_argument("--task_name", type=str, default="movie_tagging")
    return parser


def main(args):
    model_name = args.model_name
    batch_size = args.batch_size
    K = args.k

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "[PAD]"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    base_model.config.use_cache = True
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id

    with open(f"./data/{args.task_name}/user_base_LLM.json", "r") as f:
        train_data = json.load(f)

    with open(f"./data/{args.task_name}/test_100/user_test_100.json", "r") as f:
        test_data = json.load(f)

    with open(f"./data/{args.task_name}/user_anchor_candidate.json", "r") as f:
        anchor_data = json.load(f)

    with open(f"./data/{args.task_name}/user_reserve_10_percent.json", "r") as f:
        reserve_data = json.load(f)

    data = train_data + test_data + anchor_data + reserve_data
    with open("./prompt/prompt_profile.json", "r") as f:
        prompt_template = json.load(f)

    all_out = {}
    prompt_list_others = []
    userid_list_others = []

    for user in tqdm(data):
        history_list = []
        if len(user["profile"])> K:
            profiles = random.sample(user["profile"], K)
        else:
            profiles = user["profile"]

        for p in profiles:
            for key, value in p.items():
                p[key] = get_first_k_tokens(p[key], 200)

        for p in profiles:
            history_list.append(prompt_template[args.task_name]["retrieval_history"].format(**p))

        test_prompt = prompt_template[args.task_name]["profile_prompt"].format(history_list)

        prompt_list_others.append(test_prompt)
        userid_list_others.append(user["user_id"])


    batched_prompt_others = split_batch(prompt_list_others, batch_size)
    out_list_others = []

    with torch.no_grad():
        for _, batch in tqdm(enumerate(batched_prompt_others), total=len(batched_prompt_others)):
            sentences = batch
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, return_token_type_ids=False)
            inputs = inputs.to(base_model.device)

            with torch.autocast(device_type="cuda"):
                outputs = base_model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=10,
                    temperature=0.6,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=300,
                )

            out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            out_list_others += out_sentence

    pred_all_others = []

    for i in range(len(out_list_others)):
        output = out_list_others[i].replace(prompt_list_others[i], "")
        pred_all_others.append({
            "id": userid_list_others[i],
            "output": output,
        })
        all_out[userid_list_others[i]] = output

    with open(f"./data/{args.task_name}/profile-id2text.json", "w") as f:
        json.dump(all_out, f)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
