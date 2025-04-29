#!/usr/bin/env python
# Muril_preprocess_squad_v2.py

import json
import os
import torch
from tqdm import tqdm
# Instead of XLMRobertaTokenizerFast:
from transformers import BertTokenizerFast  # or AutoTokenizer

###############################################
# 1) Config
###############################################
# Adjust to your SQuAD v2.0–style JSONs
train_json = "Telugu Data/squad2.0_telugu_train.json"   # SQuAD 2.0 train
val_json   = "Telugu Data/squad2.0_telugu_val.json"     # SQuAD 2.0 val
test_json  = "Telugu Data/squad2.0_telugu_test.json"    # SQuAD 2.0 test
out_dir    = "murill_processed_telugu_squad_v2"

# train_json = "English Data/squad2.0_train.json"   # SQuAD 2.0 train
# val_json   = "English Data/squad2.0_val.json"     # SQuAD 2.0 val
# test_json  = "English Data/squad2.0_test.json"    # SQuAD 2.0 test
# out_dir    = "murill_processed_english_squad_v2"

os.makedirs(out_dir, exist_ok=True)

max_length = 512
doc_stride = 128
model_tokenizer_name = "google/muril-large-cased"  # a BERT-based model

###############################################
def load_squad_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append({
                    "id": qa["id"],
                    "context": context,
                    "question": qa["question"],
                    "is_impossible": qa.get("is_impossible", False),
                    "answers": qa.get("answers", []),
                    "plausible_answers": qa.get("plausible_answers", [])
                })
    return examples

def build_items(examples, tokenizer, max_length=384, doc_stride=128):
    data_list = []
    used_stride = min(doc_stride, max_length - 17)
    if used_stride < 0:
        used_stride = 0

    for ex in tqdm(examples, desc="Processing"):
        context = ex["context"]
        question= ex["question"]
        is_imp  = ex["is_impossible"]
        gold_text = ""
        start_char, end_char = None, None
        if (not is_imp) and ex["answers"]:
            ans = ex["answers"][0]
            gold_text  = ans["text"]
            start_char = ans["answer_start"]
            end_char   = start_char + len(gold_text)

        enc = tokenizer(
            question,
            context,
            max_length=max_length,
            truncation="only_second",
            stride=used_stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt"
        )

        for i in range(len(enc["input_ids"])):
            input_ids_i      = enc["input_ids"][i]
            attention_mask_i = enc["attention_mask"][i]
            offset_mapping_i = enc["offset_mapping"][i].tolist()

            # Find [CLS] => usually token_id=101 for BERT-based
            cls_idx = (input_ids_i == tokenizer.cls_token_id).nonzero()
            if len(cls_idx)>0:
                cls_idx = cls_idx[0].item()
            else:
                cls_idx = 0

            start_idx = cls_idx
            end_idx   = cls_idx

            if (not is_imp) and gold_text and (start_char is not None) and (end_char is not None):
                found_start = None
                found_end   = None
                for j,(off_start,off_end) in enumerate(offset_mapping_i):
                    if off_start <= start_char < off_end:
                        found_start = j
                    if off_start < end_char <= off_end:
                        found_end = j
                    if (found_start is not None) and (found_end is not None):
                        break
                if (found_start is not None) and (found_end is not None) and found_end>=found_start:
                    start_idx = found_start
                    end_idx   = found_end

            item = {
                "id": ex["id"],
                "input_ids": input_ids_i,
                "attention_mask": attention_mask_i,
                "offset_mapping": offset_mapping_i,
                "start_positions": torch.tensor(start_idx, dtype=torch.long),
                "end_positions":   torch.tensor(end_idx,   dtype=torch.long),
                "context": context,
                "gold_text": gold_text
            }
            data_list.append(item)
    return data_list

def main():
    print(f"Loading MuRIL tokenizer: {model_tokenizer_name}")
    # So we avoid the mismatch error:
    tokenizer = BertTokenizerFast.from_pretrained(model_tokenizer_name)

    print("Loading train data from:", train_json)
    train_ex = load_squad_json(train_json)
    print("Loading val data from:", val_json)
    val_ex   = load_squad_json(val_json)

    test_ex  = []
    if os.path.exists(test_json):
        print("Loading test data from:", test_json)
        test_ex = load_squad_json(test_json)
    else:
        print("No test file found => skipping")

    print("\nBuilding train items...")
    train_list = build_items(train_ex, tokenizer, max_length, doc_stride)
    print("Building val items...")
    val_list   = build_items(val_ex, tokenizer, max_length, doc_stride)

    test_list = []
    if test_ex:
        print("Building test items...")
        test_list = build_items(test_ex, tokenizer, max_length, doc_stride)

    torch.save(train_list, os.path.join(out_dir, "train.pt"))
    torch.save(val_list,   os.path.join(out_dir, "val.pt"))
    if test_list:
        torch.save(test_list, os.path.join(out_dir, "test.pt"))

    print("Done! train:", len(train_list), " val:", len(val_list), " test:", len(test_list))

if __name__ == "__main__":
    main()
