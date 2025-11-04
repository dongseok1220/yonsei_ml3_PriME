import argparse
import datetime
from functools import partial
import json
import os
from pathlib import Path
import sys 
sys.path.append("..")
import random
import time

import numpy as np
from opacus import PrivacyEngine
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from data_utils import collate_fn, InstructionDataset, process_profile
from engine_finetuning import train_one_epoch, load_model, load_generator_from_trained
from eval import evaluate_task_predictions
import llama
from utils import name2taskid, split_batch

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


def get_args_parser():
    parser = argparse.ArgumentParser("Anchor PEFT training.", add_help=False)
    parser.add_argument("--config_file", default=None, type=str, help="Config yaml file")
    parser.add_argument(
        "--batch_size",
        default=6,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--warmup_epochs", default=0, type=int)

    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters.
    parser.add_argument("--llama_model_path", default=None, type=str, help="path of llama model")
    parser.add_argument("--tokenizer_path", default=None, type=str, help="path of llama model tokenizer")

    parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")
    parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=3000, metavar="LENGTH", help="the maximum sequence length")
    parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

    # Optimizer parameters.
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (absolute lr)")
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")

    # Dataset parameters.
    parser.add_argument("--output_dir", default="./output/movie_tagging/Anchor_PEFT/LoRA", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_ckpt", default='./output/movie_tagging/task-base_LLM/lora_ckpt.pt', help="base lora")
    parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to user gradient checkpoint")
    parser.add_argument("--anchor_idx_path", default='./anchor_selection/movie_tagging/anchor_user_idx.pt', help="anchor user data")
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
    parser.add_argument("--max_gen_len", type=int, default=100, help="max generation len")
    parser.add_argument('--infer', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_eval_samples", type=int, default=None, help="no. of samples to eval on")

    # DP parameters.
    parser.add_argument("--accountant", type=str, default="rdp", help="DP accountant type")
    parser.add_argument("--sigma", type=float, default=0.60, help="Noise multiplier")
    parser.add_argument("--max_per_sample_grad_norm", type=float, default=1.0,
                        help="Clip per-sample gradients to this norm")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Target epsilon. Use the epsilon version if set; otherwise, default to the sigma version.")
    parser.add_argument("--delta", type=float, default=None, help="Target delta. If None, set to 1 / len(dataset).")
    parser.add_argument("--poisson_sample", default=True, action=argparse.BooleanOptionalAction,
                        help="Apply Poisson sampling for DP. Should be true but often gives an empty tensor esp. for a small batch."
                        "Also, gradient accumulation is forbidden.")
    parser.add_argument("--enable_dp", action="store_true", default=False,
                        help="Disable privacy training and just train with vanilla optimizer")
    parser.add_argument("--secure_rng", action="store_true", default=False,
                        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost")

    return parser


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


def create_privacy_engine(args, model, optimizer, data_loader):
    if not args.enable_dp:
        return None, model, optimizer, data_loader

    engine = PrivacyEngine(accountant=args.accountant, secure_mode=args.secure_rng)
    if args.epsilon:
        model, optimizer, data_loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_per_sample_grad_norm,
        )
    else:
        model, optimizer, data_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            poisson_sampling=args.poisson_sample,
        )
    return engine, model, optimizer, data_loader


def main(args):
    # Fix the seed for reproducibility.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.set_default_device("cuda")
    cudnn.benchmark = True
    device = torch.device(args.device)

    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write("{}".format(args).replace(", ", ",\n") + "\n")

    with open(f"./data/{args.task_name}/profile-id2text.json", "r") as f:
        all_profile = json.load(f)
    with open("./prompt/prompt.json", "r") as f:
        prompt_template = json.load(f)
    with open(args.test_data_path, 'r') as f:
        all_user_data = json.load(f)

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
    model.to(device)
    model.print_trainable_params()
    model.merge_lora_parameters()

    is_delta_set = bool(args.delta)
    anchor_idx = torch.load(args.anchor_idx_path)
    for idx in tqdm(range(len(anchor_idx))):
        idx_all_test = anchor_idx[idx]
        user = all_user_data[idx_all_test]

        user_out_dir = os.path.join(args.output_dir, "user_{}".format(user['user_id']))
        if os.path.exists(os.path.join(user_out_dir, f"train-eval.json")): continue
        Path(user_out_dir).mkdir(parents=True, exist_ok=True)

        module = model if isinstance(model, llama.model.Transformer) else model._module
        module.reset_lora_parameters()
        module.set_lora_trainable()

        data_list = process_profile(user, prompt_template, all_profile, args.task_name)
        eval_data_list = data_list
        if args.num_eval_samples:
            num_eval_samples = min(len(eval_data_list), args.num_eval_samples)
            eval_data_list = random.sample(eval_data_list, num_eval_samples)

        # Compute eval metrics prior to training.
        # A hack to use rouge for train eval in case of citation.
        task_id = "LaMP" if args.task_name in ("citation") else name2taskid[args.task_name]
        golds_dict = {"task": task_id, "golds": []}
        for datum in eval_data_list:
            prompt, full_prompt = datum["prompt"], datum["full_prompt"]
            golds_dict["golds"].append(
                {
                    "id": datum["id"],
                    "output": full_prompt.replace(prompt, "").strip(),
                }
            )
        with open(os.path.join(user_out_dir, f"train-golds.json"), "w") as f:
            json.dump(golds_dict, f, indent=4)
        generator = load_generator_from_trained(module, args.tokenizer_path)
        preds_dict = predict(generator, eval_data_list, args.task_name,
                             args.batch_size, args.max_gen_len, args.temperature, args.top_p)
        eval_results = evaluate_task_predictions(golds_dict, preds_dict)
        with open(os.path.join(user_out_dir, f"base-train-output.json"), "w") as f:
            json.dump(preds_dict, f, indent=4)
        with open(os.path.join(user_out_dir, f"base-train-eval.json"), "w") as f:
            json.dump(eval_results, f, indent=4)

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

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

        model.train(True)
        if args.epsilon:
            args.delta = args.delta if is_delta_set else 1 / len(dataset_train)
        dp_engine, model, optimizer, data_loader_train = create_privacy_engine(args, model, optimizer, data_loader_train)

        log_writer = None
        start_time = time.time()
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                model, data_loader_train, optimizer, epoch, log_writer=log_writer, args=args
            )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            if dp_engine is not None:
                epsilon = dp_engine.accountant.get_epsilon(delta=args.delta)
                log_stats.update({"dp-epsilon": epsilon, "dp-delta": args.delta})

            if args.output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(user_out_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        torch.save(module.lora_state_dict(), os.path.join(user_out_dir, "lora_ckpt.pt"))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        # Compute eval metrics on the training data.
        generator = load_generator_from_trained(module, args.tokenizer_path)
        preds_dict = predict(generator, eval_data_list, args.task_name,
                             args.batch_size, args.max_gen_len, args.temperature, args.top_p)
        eval_results = evaluate_task_predictions(golds_dict, preds_dict)
        with open(os.path.join(user_out_dir, f"train-output.json"), "w") as f:
            json.dump(preds_dict, f, indent=4)
        with open(os.path.join(user_out_dir, f"train-eval.json"), "w") as f:
            json.dump(eval_results, f, indent=4)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.test_data_path = f"./data/{args.task_name}/user_anchor_candidate.json"
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
