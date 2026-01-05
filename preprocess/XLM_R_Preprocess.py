import json
import os
import torch
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast

###############################################
# 1) Hard-coded variables
###############################################
train_json = "Telugu Data/squad2.0_telugu_train.json"   # SQuAD 2.0 train
val_json   = "Telugu Data/squad2.0_telugu_val.json"     # SQuAD 2.0 val
test_json  = "Telugu Data/squad2.0_telugu_test.json"    # SQuAD 2.0 test

out_dir    = "xlm_r_processed_telugu_squad_v2"
os.makedirs(out_dir, exist_ok=True)

max_length = 512
doc_stride = 128
model_tokenizer_name = "xlm-roberta-large"  # large recommended

###############################################
# 2) Load SQuAD v2.0
###############################################
def load_squad_json(path):
    """
    Reads a SQuAD 2.0 style JSON and returns a list of QAs.
    Each QA has:
      - "id", "context", "question", 
      - "is_impossible" (bool),
      - "answers" (list),
      - "plausible_answers" (list, optional)
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                ex = {
                    "id": qa["id"],
                    "context": context,
                    "question": qa["question"],
                    "is_impossible": qa.get("is_impossible", False),
                    "answers": qa.get("answers", []),
                    "plausible_answers": qa.get("plausible_answers", [])
                }
                examples.append(ex)
    return examples

###############################################
# 3) Build offset-based items
###############################################
def build_items(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Build items with safe document striding.
    """
    data_list = []
    stats = {
        "total": len(examples),
        "skipped": 0,
        "chunks_created": 0
    }
    
    # Minimum sequence length to allow processing
    MIN_LENGTH = 32
    
    for ex in tqdm(examples, desc="Processing"):
        question = ex["question"]
        context = ex["context"]
        is_impossible = ex["is_impossible"]
        
        # First try without striding for shorter contexts
        try:
            enc = tokenizer(
                question,
                context,
                max_length=max_length,
                truncation="only_second",
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Get default values
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]
            offset_mapping = enc["offset_mapping"][0].tolist()
            
            # Find CLS token index
            cls_indices = (input_ids == tokenizer.cls_token_id).nonzero()
            cls_idx = cls_indices[0].item() if len(cls_indices) > 0 else 0
            
            start_idx = cls_idx
            end_idx = cls_idx
            
            # Get answer info
            gold_text = ""
            start_char = None
            end_char = None
            
            if not is_impossible and ex["answers"]:
                ans = ex["answers"][0]
                gold_text = ans["text"]
                start_char = ans["answer_start"]
                end_char = start_char + len(gold_text)
                
                # Find answer span
                if start_char is not None and end_char is not None:
                    found_start = None
                    found_end = None
                    
                    for idx, (off_start, off_end) in enumerate(offset_mapping):
                        if off_start <= start_char < off_end:
                            found_start = idx
                        if off_start < end_char <= off_end:
                            found_end = idx
                        if found_start is not None and found_end is not None:
                            break
                    
                    if (found_start is not None and 
                        found_end is not None and 
                        found_end >= found_start):
                        start_idx = found_start
                        end_idx = found_end
            
            # Create item
            item = {
                "id": ex["id"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offset_mapping,
                "start_positions": torch.tensor(start_idx, dtype=torch.long),
                "end_positions": torch.tensor(end_idx, dtype=torch.long),
                "context": context,
                "gold_text": gold_text,
                "is_impossible": is_impossible
            }
            data_list.append(item)
            
        except Exception as e:
            # If first attempt fails, try with smaller chunks
            if len(context) > MIN_LENGTH:
                try:
                    # Split context into smaller chunks
                    chunk_size = max_length // 2  # Use half max_length for safety
                    chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size//2)]
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            enc = tokenizer(
                                question,
                                chunk,
                                max_length=max_length,
                                truncation="only_second",
                                return_offsets_mapping=True,
                                padding="max_length",
                                return_tensors="pt"
                            )
                            
                            input_ids = enc["input_ids"][0]
                            attention_mask = enc["attention_mask"][0]
                            offset_mapping = enc["offset_mapping"][0].tolist()
                            
                            # Default to CLS for this chunk
                            cls_indices = (input_ids == tokenizer.cls_token_id).nonzero()
                            cls_idx = cls_indices[0].item() if len(cls_indices) > 0 else 0
                            
                            item = {
                                "id": f"{ex['id']}_{i}",
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "offset_mapping": offset_mapping,
                                "start_positions": torch.tensor(cls_idx, dtype=torch.long),
                                "end_positions": torch.tensor(cls_idx, dtype=torch.long),
                                "context": chunk,
                                "gold_text": "",
                                "is_impossible": True  # Mark chunks as impossible
                            }
                            data_list.append(item)
                            stats["chunks_created"] += 1
                            
                        except Exception as chunk_e:
                            continue
                            
                except Exception as chunk_e:
                    stats["skipped"] += 1
                    continue
            else:
                stats["skipped"] += 1
                continue

    print(f"\nProcessing Statistics:")
    print(f"Total examples: {stats['total']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Additional chunks created: {stats['chunks_created']}")
    print(f"Final processed items: {len(data_list)}\n")

    return data_list

###############################################
# 4) Main function
###############################################
def main():
    print("Initializing tokenizer:", model_tokenizer_name)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_tokenizer_name)

    print("Loading train data from:", train_json)
    train_examples = load_squad_json(train_json)
    print("Loading val data from:", val_json)
    val_examples   = load_squad_json(val_json)

    test_examples  = []
    if os.path.exists(test_json):
        print("Loading test data from:", test_json)
        test_examples = load_squad_json(test_json)
    else:
        print("No separate test file found. Skipping test...")

    # Build items
    print("\nBuilding train items...")
    train_list = build_items(train_examples, tokenizer, max_length, doc_stride)
    print("Building val items...")
    val_list   = build_items(val_examples,   tokenizer, max_length, doc_stride)

    test_list = []
    if test_examples:
        print("Building test items...")
        test_list = build_items(test_examples, tokenizer, max_length, doc_stride)

    # Save
    train_out = os.path.join(out_dir, "train.pt")
    val_out   = os.path.join(out_dir, "val.pt")
    test_out  = os.path.join(out_dir, "test.pt")

    torch.save(train_list, train_out)
    torch.save(val_list,   val_out)
    if test_list:
        torch.save(test_list, test_out)

    print(f"\nSaved processed data to {out_dir}/:")
    print("  train.pt =>", len(train_list), "items")
    print("  val.pt   =>", len(val_list),   "items")
    if test_list:
        print("  test.pt  =>", len(test_list), "items")

    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()