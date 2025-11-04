import json
import zipfile
import glob
import os
os.environ['HF_EVALUATE_OFFLINE'] = '1'
import shutil
import warnings

from sklearn.exceptions import UndefinedMetricWarning

import evaluate


def get_labels(task_name):
    if task_name == "LaMP_1":
        return ["[1]", "[2]"]
    elif task_name == "LaMP_2N":
        return ["food & drink", "sports", "education", "parents", "religion", "travel", "business", "crime", "science & technology", "culture & arts", "entertainment", "politics", "women", "style & beauty", "healthy living"]
    elif task_name == "LaMP_2M":
        return ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia", "dark comedy", "classic", "psychology", "fantasy", "romance", "thought-provoking", "social commentary", "violence", "true story"]        
    elif task_name == "LaMP_3":
        return ["1", "2", "3", "4", "5"]
    else:
        raise ValueError("Invalid task_name")


def evaluate_task_predictions(golds_dict, preds_dict):
    assert golds_dict["task"] == preds_dict["task"]
    task_name = golds_dict["task"]

    golds_dict = {y["id"]: y["output"] for y in golds_dict["golds"]}
    preds_dict = {x["id"]: x["output"] for x in preds_dict["golds"]}

    gold_ids = set(golds_dict.keys())
    pred_ids = set(preds_dict.keys())
    assert gold_ids >= pred_ids, "Unexpected predictions ids exist. {}".format(pred_ids-gold_ids)

    if task_name in ["LaMP_1", "LaMP_2N", "LaMP_2M"]:
        metric = create_metric_f1_accuracy(get_labels(task_name))
    elif task_name == "LaMP_3":
        metric = create_metric_mae_rmse()
    else:
        metric = create_metric_rouge()

    pred_ids = list(pred_ids)
    golds = [golds_dict[id] for id in pred_ids]
    preds = [preds_dict[id] for id in pred_ids]
    return metric(preds, golds)


def postprocess_text_classification(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]
    return preds, labels


def postprocess_text_generation(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def create_metric_f1_accuracy(all_labels):
    f1_metric = evaluate.load("metrics/f1")
    accuracy_metric = evaluate.load("metrics/accuracy")

    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            return -1

    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        # In case of small test data for a user f1 may be ill-defined for some labels.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels,
                                          labels=list(range(len(all_labels))), average = "macro")
        result = {"accuracy" : result_acc["accuracy"], "f1" : result_f1["f1"]}
        return result

    return compute_metrics


def create_metric_mae_rmse():
    mse_metric = evaluate.load("metrics/mse")
    mae_metric = evaluate.load("metrics/mae")
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            try:
                y = float(y)
            except:
                return 2.5
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0

    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x,x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared = False)
        result = {"MAE" : result_mae["mae"], "RMSE" : result_rmse["mse"]}
        return result
    return compute_metrics


def create_metric_rouge():
    rouge_metric = evaluate.load('metrics/rouge')
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
        return result
    return compute_metrics


class LaMPEvaluation(object):
    
    def __init__(self, all_golds_zip_file_addr = None, single_gold_json_file_addr = None, extract_addr = "./tmp") -> None:
        assert all_golds_zip_file_addr or single_gold_json_file_addr, "The golds should be provided for all datasets or at least one."
        assert not (all_golds_zip_file_addr and single_gold_json_file_addr), "The golds should be provided using zip file or json file not both."
        self.tasks_golds = dict()
        self.extract_addr = extract_addr
        self.evaluate_all_is_possible = False
        if all_golds_zip_file_addr:
            os.makedirs(self.extract_addr, exist_ok=True)
            with zipfile.ZipFile(all_golds_zip_file_addr, 'r') as zobj:
                zobj.extractall(path = extract_addr)
            for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
                with open(file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
            self._empty_dir(self.extract_addr)
            self.evaluate_all_is_possible = True
        if single_gold_json_file_addr:
            with open(single_gold_json_file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
    
    def _empty_dir(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def _get_all_gold_ids(self, task_name):
        return set([sample['id'] for sample in self.tasks_golds[task_name]])
    
    def _get_all_ids(self, input):
        return set([sample['id'] for sample in input])
    
    def evaluate_all(self, predicts_zipfile_addr):
        assert self.evaluate_all_is_possible, "You did not provide golds for all tasks."
        with zipfile.ZipFile(predicts_zipfile_addr, 'r') as zobj:
            zobj.extractall(path = self.extract_addr)
        results_raw = dict()
        all_task_names = set()
        for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
            with open(file_addr) as file:
                preds = json.load(file)
            all_task_names.add(preds['task'])
            results_raw[preds['task']] = self._evaluate_task(preds['golds'], preds['task'])
        self._empty_dir(self.extract_addr)
        assert len(all_task_names) == 7, "The provided results do not cover all the tasks in the benchmark."
        return results_raw

    def evaluate_task(self, predicts_json_addr, task_name):
        with open(predicts_json_addr) as file:
            preds = json.load(file)
        assert preds['task'] == task_name, "The provided task_name and the results do not match."
        assert preds['task'] in self.tasks_golds.keys(), "The provided golds cannot be used to evaluate this task."
        return self._evaluate_task(preds['golds'], task_name)

    # TODO(kykim): Let's do some refactoring please.
    def evaluate_task_preds(self, golds_dict, preds_dict, task_name):
        gold_ids = set(golds_dict.keys())
        pred_ids = set(preds_dict.keys())
        assert gold_ids == pred_ids, "Predictions ids and gold ids do not match. {}".format(gold_ids-pred_ids)

        if task_name in ["LaMP_1", "LaMP_2N", "LaMP_2M"]:
            metric = create_metric_f1_accuracy(get_labels(task_name))
        elif task_name == "LaMP_3":
            metric = create_metric_mae_rmse()
        else:
            metric = create_metric_rouge()

        gold_ids = list(gold_ids)
        golds = [golds_dict[id] for id in gold_ids]
        preds = [preds_dict[id] for id in gold_ids]
        return metric(preds, golds)

    def _evaluate_task(self, predictions, task_name):
        golds_dict = {y['id']:y['output'] for y in self.tasks_golds[task_name]}
        preds_dict = {x['id']:x['output'] for x in predictions}
        
        gold_ids = self._get_all_gold_ids(task_name)
        pred_ids = self._get_all_ids(predictions)

        assert gold_ids == pred_ids, "Predictions ids and gold ids do not match. {}".format(gold_ids-pred_ids)

        if task_name in ["LaMP_1", "LaMP_2N", "LaMP_2M"]:
            metric = create_metric_f1_accuracy(get_labels(task_name))
        elif task_name == "LaMP_3":
            metric = create_metric_mae_rmse()
        else:
            metric = create_metric_rouge()
        
        gold_ids = list(gold_ids)
        golds = [golds_dict[id] for id in gold_ids]
        preds = [preds_dict[id] for id in gold_ids]
        return metric(preds, golds)
