import argparse
import datetime
from functools import partial
import json
import os
from pathlib import Path
import sys
sys.path.append("..")
import time

import numpy as np
from opacus import PrivacyEngine
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from data_utils import collate_fn, InstructionDataset, process_profile
from engine_finetuning import train_one_epoch, load_model
import llama

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


def get_args_parser():
    parser = argparse.ArgumentParser("Anchor gate training", add_help=False)
    parser.add_argument("--config_file", default=None, type=str, help="Config yaml file")
    parser.add_argument(
        "--batch_size",
        default=6,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--max_step", default=50, type=int)
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

    parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=3300, metavar="LENGTH", help="the maximum sequence length")
    parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

    # Optimizer parameters.
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (absolute lr)")
    parser.add_argument("--clip", type=float, default=0.3, help="gradient clipping")
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")

    # Dataset parameters.
    parser.add_argument("--output_dir", default="./output/movie_tagging/Anchor_PEFT/gate", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_ckpt", default='./output/movie_tagging/task-base_LLM/lora_ckpt.pt', help="resume lora from checkpoint")
    parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to use gradient checkpoint")
    parser.add_argument("--w_gate", type=bool, default=True, help="whether to use gate")
    
    parser.add_argument("--anchor_path", type=str, default='./output/movie_tagging/Anchor_PEFT', help="resume lora from checkpoint")
    parser.add_argument("--anchor_idx_path", type=str, default='./anchor_selection/most_active/anchor_user_idx.pt', help="resume lora from checkpoint")

    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Generation hyperparameters.
    parser.add_argument("--k_list", type=list, default=[1, 2, 4], help="top_p")
    parser.add_argument("--infer", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--recent_k", type=int, default=2000, help="top_p")

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

    anchor_idx = torch.load(args.anchor_idx_path)
    with open(args.train_data_path, "r") as f:
        all_user_data = json.load(f)
    with open("./prompt/prompt.json", "r") as f:
        prompt_template = json.load(f)
    with open(f"./data/{args.task_name}/profile-id2text.json", "r") as f:
        all_profile = json.load(f)

    # Define the model.
    model = load_model(
        ckpt_dir=args.llama_model_path,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        lora_path=args.lora_ckpt,
        w_lora=args.w_lora,
        grad_ckpt=args.grad_ckpt,
        w_gate=args.w_gate,
    )
    model.to(device)
    model.merge_lora_parameters()

    is_delta_set = bool(args.delta)
    for idx in tqdm(range(len(anchor_idx))):
        idx_all_test = anchor_idx[idx]
        user = all_user_data[idx_all_test]

        user_basename = f"user_{user['user_id']}"
        user_out_dir = os.path.join(args.output_dir, user_basename)
        Path(user_out_dir).mkdir(parents=True, exist_ok=True)

        gate_ckpt = os.path.join(user_out_dir, "gate_ckpt.pt")
        if os.path.exists(gate_ckpt):
            print(f"Gate for user {user_basename} already exists")
            continue

        lora_path = os.path.join(args.anchor_path, user_basename, "lora_ckpt.pt")
        print(lora_path)
        lora_ckpt = torch.load(lora_path, map_location="cpu")
        module = model if isinstance(model, llama.model.Transformer) else model._module
        module.load_state_dict(lora_ckpt, strict=False)
        module.reset_gate_parameters()
        module.set_gate_trainable()
        module.print_trainable_params()

        data_list = process_profile(user, prompt_template, all_profile, args.task_name)
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

        # The model needs to be in train mode to appease Opacus. However, this also turns the
        # dropout layer to training mode, which confuses the library since we're training only the
        # gates. We've instead added a hack to explicitly set the dropout layer to eval mode in
        # case of gate training in model.py.
        model.train(True)
        if args.epsilon:
            args.delta = args.delta if is_delta_set else 1 / len(dataset_train)
        args.epochs = max(int(args.max_step / (len(data_list) / args.batch_size)), 1)
        dp_engine, model, optimizer, data_loader_train = create_privacy_engine(args, model, optimizer, data_loader_train)

        epoch = 0
        args.cur_step = 0
        log_writer = None

        start_time = time.time()
        while args.cur_step < args.max_step:
            train_stats = train_one_epoch(
                model, data_loader_train, optimizer, epoch, log_writer=log_writer, args=args
            )
            epoch += 1
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

        torch.save(module.gate_state_dict(), gate_ckpt)
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.train_data_path = f"./data/{args.task_name}/user_anchor_candidate.json"
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
