import json
import math
import os
import re
import unicodedata

import numpy as np
from tqdm import tqdm

from ..process import rel2text


def save_dict(dict, name):
    if isinstance(dict, str):
        dict = eval(dict)
    with open(f"{name}.txt", "w", encoding="utf-8") as f:
        f.write(str(dict))  # dict to str


def remove_stress_mark(text):
    text = "".join(
        [
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        ]
    )
    return text


def change_case(str):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", str)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return re.sub(r"[^\w\s]", "", s2)


# Driver code
class UniRelDataProcessor(object):
    def __init__(
        self,
        root,
        tokenizer,
        dataset_name="nyt",
    ):
        self.task_data_dir = os.path.join(root, dataset_name)
        self.train_path = os.path.join(self.task_data_dir, "train_split.json")
        self.dev_path = os.path.join(self.task_data_dir, "valid_data.json")
        self.test_path = os.path.join(self.task_data_dir, "test_data.json")

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        self.label_map_cache_path = os.path.join(
            self.task_data_dir, dataset_name + ".dict"
        )

        self.label2id = None
        self.id2label = None
        self.max_label_len = 0

        self._get_labels()
        if dataset_name.startswith("nyt"):
            self.pred2text = rel2text.nyt_rel2text
        elif dataset_name == "retacred":
            self.pred2text = rel2text.retacred_rel2text
        elif dataset_name.startswith("webnlg"):

            if dataset_name == "webnlg_star":
                self.pred2text = {}
                for pred in self.label2id.keys():
                    try:
                        self.pred2text[pred] = rel2text.webnlg_rel2text[pred]
                    except KeyError:
                        print(pred)
            else:
                self.pred2text = rel2text.webnlg_rel2text

            cnt = 1
            exist_value = []
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]"
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)

        self.num_rels = len(self.pred2text.keys())
        self.max_label_len = 1
        self.pred2idx = {}
        idx = 0
        self.pred_str = ""
        for k in self.pred2text:
            self.pred2idx[k] = idx
            self.pred_str += self.pred2text[k] + " "
            idx += 1
        self.pred_str = self.pred_str[:-1]
        self.idx2pred = {value: key for key, value in self.pred2idx.items()}
        self.num_labels = self.num_rels

    def get_train_sample(self, token_len=100, data_nums=-1):
        return self._pre_process(
            self.train_path, token_len=token_len, data_nums=data_nums
        )

    def get_dev_sample(self, token_len=150, data_nums=-1):
        return self._pre_process(
            self.dev_path, token_len=token_len, data_nums=data_nums
        )

    def get_test_sample(self, token_len=150, data_nums=-1):
        samples = self._pre_process(
            self.test_path, token_len=token_len, data_nums=data_nums
        )
        return samples

    def get_specific_test_sample(self, data_path, token_len=150, data_nums=-1):
        return self._pre_process(
            data_path, token_len=token_len, data_nums=data_nums
        )

    def _get_labels(self):
        label_num_dict = {}
        label_set = set()
        for path in [self.train_path, self.dev_path, self.test_path]:
            fp = open(path)
            samples = json.load(fp)
            for data in samples:
                sample = data
                for spo in sample["relation_list"]:
                    label_set.add(spo["predicate"])
                    if spo["predicate"] not in label_num_dict:
                        label_num_dict[spo["predicate"]] = 0
                    label_num_dict[spo["predicate"]] += 1
        label_set = sorted(label_set)
        labels = list(label_set)
        label_map = {idx: label for idx, label in enumerate(labels)}
        self.id2label = label_map
        self.label2id = {val: key for key, val in self.id2label.items()}

    def _pre_process(self, path, token_len, data_nums):
        outputs = {
            "text": [],
            "spo_list": [],
            "spo_span_list": [],
            "head_label": [],
            "tail_label": [],
            "span_label": [],
        }
        token_len_big_than_100 = 0
        token_len_big_than_150 = 0
        max_token_len = 0
        max_data_nums = math.inf if data_nums == -1 else data_nums
        data_count = 0
        data = json.load(open(path))
        label_dict = {}
        for line in tqdm(data):
            if len(line["relation_list"]) == 0:
                continue
            text = line["text"]
            input_ids = self.tokenizer.encode(text)
            token_encode_len = len(input_ids)
            if token_encode_len > 100 + 2:
                token_len_big_than_100 += 1
            if token_encode_len > 150 + 2:
                token_len_big_than_150 += 1
            max_token_len = max(max_token_len, token_encode_len)
            if token_encode_len > token_len + 2:
                continue
            spo_list = set()
            spo_span_list = set()
            # [CLS] texts [SEP] rels
            head_matrix = np.zeros(
                [token_len + 2 + self.num_rels, token_len + 2 + self.num_rels]
            )
            tail_matrix = np.zeros(
                [token_len + 2 + self.num_rels, token_len + 2 + self.num_rels]
            )
            span_matrix = np.zeros(
                [token_len + 2 + self.num_rels, token_len + 2 + self.num_rels]
            )

            e2e_set = set()
            spo_tail_set = set()
            spo_tail_text_set = set()
            spo_text_set = set()
            for spo in line["relation_list"]:
                pred = spo["predicate"]
                if pred not in label_dict:
                    label_dict[pred] = 0
                label_dict[pred] += 1
                sub = spo["subject"]
                obj = spo["object"]
                spo_list.add((sub, pred, obj))
                sub_span = spo["subj_tok_span"]
                obj_span = spo["obj_tok_span"]
                pred_idx = self.pred2idx[pred]
                plus_token_pred_idx = pred_idx + token_len + 2
                spo_span_list.add((tuple(sub_span), pred_idx, tuple(obj_span)))

                h_s, h_e = sub_span
                t_s, t_e = obj_span
                # Entity-Entity Interaction
                head_matrix[h_s + 1][t_s + 1] = 1
                head_matrix[t_s + 1][h_s + 1] = 1
                tail_matrix[h_e][t_e] = 1
                tail_matrix[t_e][h_e] = 1
                span_matrix[h_s + 1][h_e] = 1
                span_matrix[h_e][h_s + 1] = 1
                span_matrix[t_s + 1][t_e] = 1
                span_matrix[t_e][t_s + 1] = 1
                # Subject-Relation Interaction
                head_matrix[h_s + 1][plus_token_pred_idx] = 1
                tail_matrix[h_e][plus_token_pred_idx] = 1
                span_matrix[h_s + 1][plus_token_pred_idx] = 1
                span_matrix[h_e][plus_token_pred_idx] = 1
                span_matrix[t_s + 1][plus_token_pred_idx] = 1
                span_matrix[t_e][plus_token_pred_idx] = 1
                # Relation-Object Interaction
                head_matrix[plus_token_pred_idx][t_s + 1] = 1
                tail_matrix[plus_token_pred_idx][t_e] = 1
                span_matrix[plus_token_pred_idx][t_s + 1] = 1
                span_matrix[plus_token_pred_idx][t_e] = 1
                span_matrix[plus_token_pred_idx][h_s + 1] = 1
                span_matrix[plus_token_pred_idx][h_e] = 1

                spo_tail_set.add((h_e, plus_token_pred_idx, t_e))
                spo_tail_text_set.add(
                    (
                        self.tokenizer.decode(input_ids[h_e]),
                        pred,
                        self.tokenizer.decode(input_ids[t_e]),
                    )
                )
                spo_text_set.add(
                    (
                        self.tokenizer.decode(input_ids[h_s + 1 : h_e + 1]),
                        pred,
                        self.tokenizer.decode(input_ids[t_s + 1 : t_e + 1]),
                    )
                )
                e2e_set.add((h_e, t_e))
                e2e_set.add((t_e, h_e))

            outputs["text"].append(text)
            outputs["spo_list"].append(list(spo_list))
            outputs["spo_span_list"].append(list(spo_span_list))
            outputs["head_label"].append(head_matrix)
            outputs["tail_label"].append(tail_matrix)
            outputs["span_label"].append(span_matrix)

            data_count += 1
            if data_count >= max_data_nums:
                break

        print(max_token_len)
        print(f"more than 100: {token_len_big_than_100}")
        print(f"more than 150: {token_len_big_than_150}")
        return outputs
