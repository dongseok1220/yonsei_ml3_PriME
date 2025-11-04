"""Script to combine json result files from merging runs."""

import argparse
import glob
import json
import os

from eval import evaluate_task_predictions
from utils import name2taskid


def get_args_parser():
    parser = argparse.ArgumentParser("Post-processing of merging results.", add_help=False)
    parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")
    parser.add_argument("--output_dir", default="./output/movie_tagging/evo-LoRA-Composition", help="path where to save, empty for no saving")
    return parser


def main(args):
    print(f"Start postprocessing: {args.output_dir}")

    # Not ideal, as it needs to be kept consistent with the merging scripts but do sth quick for now.
    tmp_dir = os.path.join(args.output_dir, "tmp")

    # Read all prediction files.
    pred_dicts_read, pred_profile_dicts_read = [], []
    for fname_pattern, out_data in zip(
        ["*-pred.json", "*-pred-profile.json"],
        [pred_dicts_read, pred_profile_dicts_read],
    ):
        for fname in glob.glob(os.path.join(tmp_dir, fname_pattern)):
            with open(fname, "r") as f:
                out_data.append(json.load(f))

    with open(args.test_gold_data_path, "r") as f:
        test_user_golds = json.load(f)

    k_list = list(pred_dicts_read[0].keys())

    # Merge predictions and evaluate.
    # Note that json treats keys as strings and hence str(idx).
    for k in k_list:
        pred_all_out = {"task": name2taskid[args.task_name], "golds": []}
        for pred_dict in pred_dicts_read:
            for user_pred in pred_dict[k].values():
                pred_all_out["golds"].extend(user_pred)

        pred_profile_all_out = {"task": name2taskid[args.task_name], "golds": []}
        for pred_profile_dict in pred_profile_dicts_read:
            for user_pred in pred_profile_dict[k].values():
                pred_profile_all_out["golds"].extend(user_pred)

        with open(os.path.join(args.output_dir, f"output-k{k}.json"), "w") as f:
            json.dump(pred_all_out, f, indent=4)
        with open(os.path.join(args.output_dir, f"output-k{k}-profile.json"), "w") as f:
            json.dump(pred_profile_all_out, f, indent=4)

        pred_eval = evaluate_task_predictions(test_user_golds, pred_all_out)
        pred_profile_eval = evaluate_task_predictions(test_user_golds, pred_profile_all_out)
        with open(os.path.join(args.output_dir, f"eval-k{k}.json"), "w") as f:
            json.dump(pred_eval, f, indent=4)
        with open(os.path.join(args.output_dir, f"eval-k{k}-profile.json"), "w") as f:
            json.dump(pred_profile_eval, f, indent=4)

    print(f"Done postprocessing: {args.output_dir}")


if __name__ == "__main__":
    repo_base = os.path.dirname(os.path.realpath(__name__))
    args = get_args_parser()
    args = args.parse_args()
    args.test_gold_data_path = f"./data/{args.task_name}/test_100/user_test_100_label.json"
    print(args)
    main(args)
