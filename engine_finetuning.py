import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import torch

from llama import LLaMA, Llama3, ModelArgs, Tokenizer, Tokenizer3, Transformer
import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (examples, labels, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # DP with Poisson sampling often returns empty tensors.
        if examples.nelement() < 1: continue

        examples = examples.cuda()
        labels = labels.cuda()
        
        c_loss = model(examples, labels)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss.backward()

        if (data_iter_step + 1) % accum_iter == 0:
            if args.clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            # We use epoch_1000x as the x-axis in tensorboard.
            # This calibrates different curves when batch size changes.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
        
        if getattr(args, "max_step", None) is not None:
            args.cur_step += 1
            if args.cur_step+1 >= args.max_step:
                break

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _is_llama3(model_path: str):
    model_names = ["Llama3.1", "Llama3.2", "Llama3.3"]
    return any(model_name in model_path for model_name in model_names)


def get_tokenizer(tokenizer_path: str):
    tokenizer_cls = Tokenizer3 if _is_llama3(tokenizer_path) else Tokenizer
    tokenizer = tokenizer_cls(model_path=tokenizer_path + "/tokenizer.model")
    return tokenizer


def get_llama(model, tokenizer):
    llama_cls = Llama3 if isinstance(tokenizer, Tokenizer3) else LLaMA
    generator = llama_cls(model, tokenizer)
    return generator


def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    lora_path: str = None,
    w_lora: bool = False,
    grad_ckpt: bool = True,
    w_gate: bool  =False,
    target_modules = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
):
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = checkpoints[0]

    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if lora_path is not None:
        adapter_checkpoint = torch.load(lora_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        w_lora=w_lora,
        grad_ckpt=grad_ckpt,
        w_gate=w_gate,
        target_modules=target_modules,
        **params,
    )
    tokenizer = get_tokenizer(tokenizer_path)

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)

    model = Transformer(model_args)
    model.eval()
    model.train(False)

    model.load_state_dict(checkpoint, strict=False)

    if lora_path is not None:
        model.load_state_dict(adapter_checkpoint, strict=False)

    model = model.cuda()
    if w_lora:
        model.set_lora_trainable()

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model


def load_generator_from_trained(
    model,
    tokenizer_path: str,
) -> LLaMA | Llama3:
    tokenizer = get_tokenizer(tokenizer_path)
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)

    model.eval()
    model.train(False)

    model = model.cuda()
    generator = get_llama(model, tokenizer)
    return generator
