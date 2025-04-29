#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

def convert_to_squad_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    squad_data = {
        "version": "v2.0",
        "data": []
    }

    for article in data["data"]:
        article_data = {
            "title": article.get("title", ""),
            "paragraphs": []
        }

        for paragraph in article["paragraphs"]:
            paragraph_data = {
                "context": paragraph["context"],
                "qas": []
            }

            for qa in paragraph["qas"]:
                # Check if the question is impossible (i.e., has an empty answer)
                if qa["answers"][0]["text"] == "":
                    # Impossible question
                    qas = {
                        "id": qa["id"],
                        "question": qa["question"],
                        "answers": [{"text": "", "answer_start": None}],
                        "is_impossible": True
                    }
                else:
                    # Answerable question
                    qas = {
                        "id": qa["id"],
                        "question": qa["question"],
                        "answers": [{"text": qa["answers"][0]["text"], "answer_start": qa["answers"][0]["answer_start"]}],
                        "is_impossible": False
                    }
                
                paragraph_data["qas"].append(qas)
            article_data["paragraphs"].append(paragraph_data)

        squad_data["data"].append(article_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(squad_data, f, ensure_ascii=False, indent=4)

# Example usage:
input_file = "indicqa.te.json"  # Path to the original IndicQA.TE file
output_file = "indicqa_squad_format.json"  # Path to save the converted SQuAD 2.0 file
convert_to_squad_format(input_file, output_file)


# In[14]:


#!/usr/bin/env python
import json
import csv
from tqdm import tqdm
import re

CHUNK_SIZE = 500
OVERLAP_SIZE = 100  # Characters of overlap between chunks

def assign_context_ids(input_path, output_path, contexts_csv_path):
    """
    Step 1: Assign unique IDs to contexts and save both modified JSON and CSV
    """
    print(f"Reading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    csv_rows = []
    context_id = 1
    modified_data = {"version": data["version"], "data": []}
    
    print("Assigning context IDs...")
    for article in data["data"]:
        new_article = {
            "title": article.get("title", ""),
            "paragraphs": []
        }
        
        for para in article["paragraphs"]:
            cid = f"c_{context_id:03d}"
            
            csv_rows.append({
                "context_id": cid,
                "context": para["context"]
            })
            
            new_para = {
                "context": para["context"],
                "context_id": cid,
                "qas": para["qas"]
            }
            new_article["paragraphs"].append(new_para)
            context_id += 1
        
        modified_data["data"].append(new_article)
    
    print(f"Writing contexts CSV to {contexts_csv_path}")
    with open(contexts_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["context_id", "context"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"Writing modified JSON to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {context_id-1} contexts")
    return modified_data

def find_word_boundary(text, pos, forward=True):
    """
    Find the nearest word boundary from pos, moving either forward or backward
    """
    if forward:
        space_pos = text.find(' ', pos)
        return len(text) if space_pos == -1 else space_pos
    else:
        space_pos = text.rfind(' ', 0, pos)
        return 0 if space_pos == -1 else space_pos + 1

def split_context(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE):
    """
    Split context into overlapping chunks at word boundaries
    """
    if len(text) <= chunk_size:
        return [(0, text)]
    
    chunks = []
    pos = 0
    
    while pos < len(text):
        # Find end of current chunk (with overlap)
        chunk_end = min(pos + chunk_size, len(text))
        
        # Extend to word boundary
        if chunk_end < len(text):
            chunk_end = find_word_boundary(text, chunk_end)
        
        # Extract chunk
        chunk = text[pos:chunk_end]
        chunks.append((pos, chunk))
        
        # Move to next position (with overlap)
        if chunk_end >= len(text):
            break
            
        # Start next chunk at a word boundary before current chunk end
        pos = find_word_boundary(text, chunk_end - overlap, forward=False)
        
        # Prevent infinite loop
        if pos >= chunk_end:
            pos = chunk_end
    
    return chunks

def create_context_splits(json_with_ids_path, splits_output_path):
    """
    Step 2: Create context splits JSON with sub-context IDs
    """
    print(f"Reading JSON with context IDs: {json_with_ids_path}")
    with open(json_with_ids_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    splits_map = {}
    
    print("Creating context splits...")
    for article in tqdm(data["data"]):
        for para in article["paragraphs"]:
            context = para["context"]
            cid = para["context_id"]
            
            # Split context into overlapping chunks
            chunks = split_context(context)
            
            # Create sub-contexts with IDs
            sub_contexts = []
            for i, (start_pos, chunk_text) in enumerate(chunks, 1):
                sub_id = f"{cid}_{i:02d}"
                sub_contexts.append({
                    "id": sub_id,
                    "text": chunk_text,
                    "start_idx": start_pos,
                    "original_context": context
                })
            
            splits_map[cid] = sub_contexts
    
    print(f"Writing splits to {splits_output_path}")
    with open(splits_output_path, 'w', encoding='utf-8') as f:
        json.dump(splits_map, f, ensure_ascii=False, indent=2)
    
    total_contexts = len(splits_map)
    total_subcontexts = sum(len(splits) for splits in splits_map.values())
    avg_splits = total_subcontexts / total_contexts if total_contexts > 0 else 0
    
    print("\nProcessing complete!")
    print(f"Total contexts processed: {total_contexts}")
    print(f"Total sub-contexts created: {total_subcontexts}")
    print(f"Average splits per context: {avg_splits:.1f}")

def main():
    input_json = "indicqa_squad_format.json"
    json_with_ids = "indicqa_squad_with_ids.json"
    contexts_csv = "contexts.csv"
    splits_json = "context_splits.json"
    
    print("\nStep 1: Assigning Context IDs")
    print("=" * 50)
    assign_context_ids(input_json, json_with_ids, contexts_csv)
    
    print("\nStep 2: Creating Context Splits")
    print("=" * 50)
    create_context_splits(json_with_ids, splits_json)
    
    print("\nAll steps completed successfully!")
    print(f"1. JSON with context IDs: {json_with_ids}")
    print(f"2. Contexts CSV: {contexts_csv}")
    print(f"3. Context splits JSON: {splits_json}")

if __name__ == "__main__":
    main()


# In[16]:


#!/usr/bin/env python
import json
from tqdm import tqdm

def find_answer_in_context(text, answer_text):
    """Find all occurrences of answer_text in text"""
    positions = []
    start = 0
    while True:
        pos = text.find(answer_text, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions

def find_subcontext_for_answer(sub_contexts, current_idx, answer_text):
    """
    Try to find answer in current, next, and previous sub-contexts.
    Returns (sub_context_id, new_answer_start) or (None, None) if not found.
    """
    # First check current context
    current = sub_contexts[current_idx]
    positions = find_answer_in_context(current['text'], answer_text)
    if positions:
        return current['id'], positions[0]
    
    # Check next context if available
    if current_idx + 1 < len(sub_contexts):
        next_ctx = sub_contexts[current_idx + 1]
        positions = find_answer_in_context(next_ctx['text'], answer_text)
        if positions:
            return next_ctx['id'], positions[0]
    
    # Check previous context if available
    if current_idx > 0:
        prev_ctx = sub_contexts[current_idx - 1]
        positions = find_answer_in_context(prev_ctx['text'], answer_text)
        if positions:
            return prev_ctx['id'], positions[0]
    
    return None, None

def construct_qa_dataset(original_squad_path, subcontexts_json_path, output_path):
    """
    Construct QA dataset checking adjacent contexts for answers.
    """
    print(f"Loading original data from {original_squad_path}")
    with open(original_squad_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"Loading sub-contexts from {subcontexts_json_path}")
    with open(subcontexts_json_path, 'r', encoding='utf-8') as f:
        contexts_map = json.load(f)
    
    new_data = {
        "version": "v2.0_processed",
        "data": []
    }
    
    total_questions = 0
    questions_placed = 0
    lost_questions = []
    
    print("Processing QA pairs...")
    for article in tqdm(original_data['data']):
        new_article = {
            "title": article.get('title', ''),
            "paragraphs": []
        }
        
        for para in article['paragraphs']:
            context = para['context']
            qas = para['qas']
            total_questions += len(qas)
            
            cid = para.get('context_id')
            if not cid or cid not in contexts_map:
                continue
            
            sub_contexts = contexts_map[cid]
            subcontext_qas = {}
            
            # Calculate rough position for each sub-context
            sub_context_positions = []
            current_pos = 0
            for sub_ctx in sub_contexts:
                sub_context_positions.append(current_pos)
                current_pos += len(sub_ctx['text'])
            
            for qa in qas:
                if qa.get('is_impossible', False):
                    # Add unanswerable questions to first sub-context
                    sc_id = sub_contexts[0]['id']
                    if sc_id not in subcontext_qas:
                        subcontext_qas[sc_id] = []
                    new_qa = qa.copy()
                    new_qa['answers'] = []
                    subcontext_qas[sc_id].append(new_qa)
                    questions_placed += 1
                    continue
                
                if not qa.get('answers'):
                    continue
                
                answer = qa['answers'][0]
                ans_start = answer.get('answer_start')
                ans_text = answer.get('text')
                
                if ans_start is None or ans_text is None:
                    continue
                
                # Find which sub-context is closest to the answer position
                closest_idx = 0
                closest_dist = float('inf')
                for idx, pos in enumerate(sub_context_positions):
                    dist = abs(ans_start - pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = idx
                
                # Try to find answer in current/adjacent sub-contexts
                sc_id, new_start = find_subcontext_for_answer(
                    sub_contexts, closest_idx, ans_text
                )
                
                if sc_id is not None:
                    if sc_id not in subcontext_qas:
                        subcontext_qas[sc_id] = []
                    
                    new_qa = qa.copy()
                    new_qa['answers'] = [answer.copy()]
                    new_qa['answers'][0]['answer_start'] = new_start
                    subcontext_qas[sc_id].append(new_qa)
                    questions_placed += 1
                else:
                    # If still not found, try all sub-contexts
                    found = False
                    for idx in range(len(sub_contexts)):
                        sc_id, new_start = find_subcontext_for_answer(
                            sub_contexts, idx, ans_text
                        )
                        if sc_id is not None:
                            if sc_id not in subcontext_qas:
                                subcontext_qas[sc_id] = []
                            new_qa = qa.copy()
                            new_qa['answers'] = [answer.copy()]
                            new_qa['answers'][0]['answer_start'] = new_start
                            subcontext_qas[sc_id].append(new_qa)
                            questions_placed += 1
                            found = True
                            break
                    
                    if not found:
                        lost_questions.append({
                            'id': qa['id'],
                            'context_id': cid,
                            'answer_start': ans_start,
                            'answer_text': ans_text,
                            'question': qa['question']
                        })
            
            # Create new paragraphs for each sub-context that has questions
            for sub_ctx in sub_contexts:
                sc_id = sub_ctx['id']
                if sc_id in subcontext_qas:
                    new_article['paragraphs'].append({
                        "context": sub_ctx['text'],
                        "context_id": sc_id,
                        "qas": subcontext_qas[sc_id]
                    })
        
        if new_article['paragraphs']:
            new_data['data'].append(new_article)
    
    print(f"Writing processed dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    if lost_questions:
        lost_questions_path = 'lost_questions.json'
        print(f"Writing lost questions report to {lost_questions_path}")
        with open(lost_questions_path, 'w', encoding='utf-8') as f:
            json.dump(lost_questions, f, ensure_ascii=False, indent=2)
    
    print("\nProcessing complete!")
    print(f"Total questions: {total_questions}")
    print(f"Questions placed: {questions_placed}")
    print(f"Questions lost: {len(lost_questions)}")
    print(f"Success rate: {questions_placed/total_questions*100:.1f}%")
    
    if lost_questions:
        print("\nWarning: Some questions could not be placed.")
        print("Check lost_questions.json for details.")

if __name__ == "__main__":
    original_squad_path = "indicqa_squad_with_ids.json"
    subcontexts_json_path = "context_splits.json"
    output_path = "indicqa_squad_window.json"
    
    construct_qa_dataset(original_squad_path, subcontexts_json_path, output_path)


# In[18]:


#!/usr/bin/env python
import json
from tqdm import tqdm

def validate_answer_span(context, answer_text, answer_start):
    """
    Validate if the answer text matches at the given position in context.
    Returns (is_valid, details) tuple.
    """
    # Basic validation
    if answer_start < 0 or answer_start >= len(context):
        return False, "Answer start position out of context bounds"
    
    if answer_start + len(answer_text) > len(context):
        return False, "Answer text extends beyond context bounds"
    
    # Extract text at the specified position
    text_at_position = context[answer_start:answer_start + len(answer_text)]
    
    # Check if extracted text matches answer text
    if text_at_position != answer_text:
        # Search for the answer text in the context
        real_pos = context.find(answer_text)
        if real_pos != -1:
            return False, f"Answer text found at position {real_pos}, not at {answer_start}"
        return False, f"Answer text not found. At position {answer_start} found '{text_at_position}' instead of '{answer_text}'"
    
    return True, "Valid answer span"

def validate_squad_file(input_path):
    """
    Validate all answer spans in a SQuAD format file.
    """
    print(f"Reading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Statistics
    total_questions = 0
    valid_spans = 0
    invalid_spans = []
    
    print("Validating answer spans...")
    for article in tqdm(data['data']):
        for para in article['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                total_questions += 1
                
                if qa.get('is_impossible', False):
                    valid_spans += 1
                    continue
                
                if not qa.get('answers'):
                    continue
                
                answer = qa['answers'][0]
                ans_start = answer.get('answer_start')
                ans_text = answer.get('text')
                
                if ans_start is None or ans_text is None:
                    invalid_spans.append({
                        'id': qa['id'],
                        'context_id': para.get('context_id', 'unknown'),
                        'error': 'Missing answer start or text',
                        'question': qa['question']
                    })
                    continue
                
                is_valid, details = validate_answer_span(context, ans_text, ans_start)
                
                if is_valid:
                    valid_spans += 1
                else:
                    invalid_spans.append({
                        'id': qa['id'],
                        'context_id': para.get('context_id', 'unknown'),
                        'question': qa['question'],
                        'answer_text': ans_text,
                        'answer_start': ans_start,
                        'error': details,
                        'context_snippet': context[max(0, ans_start-50):min(len(context), ans_start + len(ans_text) + 50)]
                    })
    
    # Print statistics
    print("\nValidation complete!")
    print(f"Total questions: {total_questions}")
    print(f"Valid spans: {valid_spans}")
    print(f"Invalid spans: {len(invalid_spans)}")
    print(f"Validity rate: {valid_spans/total_questions*100:.1f}%")
    
    if invalid_spans:
        output_path = 'invalid_spans.json'
        print(f"\nWriting invalid spans to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(invalid_spans, f, ensure_ascii=False, indent=2)
        
        # Print some examples of invalid spans
        print("\nExample invalid spans:")
        for span in invalid_spans[:3]:  # Show first 3 examples
            print(f"\nQuestion ID: {span['id']}")
            print(f"Context ID: {span['context_id']}")
            print(f"Question: {span['question']}")
            print(f"Answer text: {span['answer_text']}")
            print(f"Error: {span['error']}")
            print(f"Context snippet: ...{span['context_snippet']}...")

def check_specific_qa(context, qa):
    """
    Check a specific QA pair and print detailed analysis.
    """
    if qa.get('is_impossible', False):
        print("This is marked as an impossible question")
        return
    
    answer = qa['answers'][0]
    ans_start = answer['answer_start']
    ans_text = answer['text']
    
    print(f"\nAnalyzing answer span for question ID: {qa['id']}")
    print(f"Question: {qa['question']}")
    print(f"Expected answer: '{ans_text}' at position {ans_start}")
    
    # Validate the span
    is_valid, details = validate_answer_span(context, ans_text, ans_start)
    print(f"\nValidation result: {'✓ Valid' if is_valid else '✗ Invalid'}")
    print(f"Details: {details}")
    
    # Show context snippet
    start_idx = max(0, ans_start - 50)
    end_idx = min(len(context), ans_start + len(ans_text) + 50)
    print("\nContext snippet:")
    print(f"...{context[start_idx:end_idx]}...")
    
    # If invalid, try to find the correct position
    if not is_valid:
        positions = []
        start = 0
        while True:
            pos = context.find(ans_text, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if positions:
            print("\nFound answer text at positions:", positions)
            for pos in positions:
                print(f"\nSnippet for position {pos}:")
                print(f"...{context[max(0, pos-20):min(len(context), pos+len(ans_text)+20)]}...")
        else:
            print("\nAnswer text not found in context")

if __name__ == "__main__":
    # For validating entire file
    validate_squad_file("indicqa_squad_window.json")
    
    # For checking specific QA pair (from your example)
    # example_context = "చేసుకున్నాడ్య్. ఉన్నత పాఠశాల విద్య ముగించాక, నారాయణ్ విశ్వవిద్యాలయ ప్రవేశ పరీక్ష తప్పి, ఇంటిలోనే చదువుకుంటూ, రాసుకుంటూ ఒక సంవత్సరం గడిపి, తర్వాత 1926 సంవత్సరము పరీక్షలో ఉత్తీర్ణుడై మైసూరు మహారాజ కళాశాలలో చేరాడు. బేచలర్ పట్టా పొందడానికి నారాయణ్ నాలుగు సంవత్సరాలు తీసుకున్నాడు. ఇది మామూలు కంటే ఒక ఏడాది ఎక్కువ. మాస్టర్ డిగ్రీ (M. A. ) చదవడం వల్ల సాహిత్యంలో ఉన్న ఆసక్తి తగ్గిపోతుందని ఒక మిత్రుడు చెప్పడంతో, కొంత కాలం ఇతడు ఒక పాఠశాల ఉపాధ్యాయునిగా ఉద్యోగం చేసాడు. అయితే, ప్రధానోపాధ్యాయుడు ఇతడిని వ్యాయామ ఉపాధ్యాయుని"
    # example_qa = {
    #     "id": 0,
    #     "question": "మాస్టర్ డిగ్రీ (M. A. ) చదవడం వల్ల సాహిత్యంలో ఉన్న ఆసక్తి తగ్గిపోతుందని మిత్రుడు చెప్పడంతో, కొంత కాలం నారాయణ్ ఏ ఉద్యోగం చేసాడు?",
    #     "answers": [
    #         {
    #             "text": "ఉపాధ్యాయునిగా",
    #             "answer_start": 429
    #         }
    #     ],
    #     "is_impossible": False
    # }
    
    # check_specific_qa(example_context, example_qa)


# ### XLM-R

# In[ ]:


#!/usr/bin/env python
import json
import os
import torch
from transformers import XLMRobertaTokenizerFast
from tqdm import tqdm

# ------------------
# Adjust paths here
# ------------------
input_json = "indicqa_squad_window.json"
out_dir = "processed_indicqa"
os.makedirs(out_dir, exist_ok=True)

output_pt = os.path.join(out_dir, "indicqa_windowed.pt")

# ------------------
# Hyperparameters
# ------------------
max_length = 512
model_tokenizer_name = "xlm-roberta-large"

def filter_and_verify_squad(input_path):
    """
    Returns a new SQuAD JSON dict containing only QAs where:
    - is_impossible=False
    - at least one answer exists
    """
    print(f"Loading data from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {
        "version": data.get("version", "windowed_indicqa"),
        "data": []
    }
    
    total_qas = 0
    valid_qas = 0
    
    print("Filtering and verifying QA pairs...")
    for article in tqdm(data["data"]):
        new_paragraphs = []
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            new_qas = []
            
            for qa in paragraph["qas"]:
                total_qas += 1
                # Skip impossible questions
                if qa.get("is_impossible", True):
                    continue
                    
                # Must have answers
                if not qa.get("answers"):
                    continue
                
                answer = qa["answers"][0]
                ans_text = answer.get("text")
                
                # Only verify answer text exists somewhere in context
                if ans_text and ans_text in context:
                    new_qas.append(qa)
                    valid_qas += 1
            
            if new_qas:
                new_paragraphs.append({
                    "context": context,
                    "context_id": paragraph.get("context_id", ""),
                    "qas": new_qas
                })
        
        if new_paragraphs:
            new_data["data"].append({
                "title": article.get("title", ""),
                "paragraphs": new_paragraphs
            })
    
    print(f"\nTotal QAs: {total_qas}")
    print(f"Valid QAs: {valid_qas}")
    print(f"Filtered out: {total_qas - valid_qas}")
    
    return new_data

def build_examples(squad_data, tokenizer, max_length=512):
    """
    Build tokenized examples for each QA pair.
    More lenient handling of answer spans for evaluation purposes.
    """
    examples_out = []
    print("Building tokenized examples...")
    
    for article in tqdm(squad_data["data"]):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_id = paragraph.get("context_id", "")
            
            for qa in paragraph["qas"]:
                ans = qa["answers"][0]
                ans_text = ans["text"]
                ans_start = ans.get("answer_start", context.find(ans_text))  # Fallback to first occurrence
                
                # If answer_start is invalid, find the first occurrence
                if ans_start < 0 or ans_start >= len(context) or context[ans_start:ans_start + len(ans_text)] != ans_text:
                    ans_start = context.find(ans_text)
                
                # If still can't find the answer, skip this example
                if ans_start == -1:
                    continue
                    
                ans_end = ans_start + len(ans_text)
                
                # Tokenize
                enc = tokenizer(
                    qa["question"],
                    context,
                    max_length=max_length,
                    truncation="only_second",
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    padding="max_length"
                )
                
                input_ids = enc["input_ids"][0]
                attention_mask = enc["attention_mask"][0]
                offset_mapping = enc["offset_mapping"][0].tolist()
                
                # Find token indices for answer span
                start_token, end_token = None, None
                for i, (start_char, end_char) in enumerate(offset_mapping):
                    if start_char <= ans_start < end_char:
                        start_token = i
                    if start_char < ans_end <= end_char:
                        end_token = i
                
                # If can't find exact token spans, use approximation
                if start_token is None or end_token is None or end_token < start_token:
                    # Find the best approximate token positions
                    for i, (start_char, end_char) in enumerate(offset_mapping):
                        if end_char > 0:  # Skip special tokens
                            start_token = i
                            break
                    
                    for i, (start_char, end_char) in enumerate(offset_mapping[start_token:], start_token):
                        if i >= len(offset_mapping) - 1 or offset_mapping[i+1][0] > ans_start + len(ans_text):
                            end_token = i
                            break
                    
                    if end_token is None:
                        end_token = start_token
                
                # Create example
                ex_item = {
                    "id": qa["id"],
                    "context_id": context_id,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "start_positions": torch.tensor(start_token, dtype=torch.long),
                    "end_positions": torch.tensor(end_token, dtype=torch.long),
                    "offset_mapping": offset_mapping,
                    "context": context,
                    "question": qa["question"],
                    "gold_text": ans_text,
                    "answer_start": ans_start  # Keep original position for evaluation
                }
                examples_out.append(ex_item)
    
    print(f"\nTotal examples built: {len(examples_out)}")
    return examples_out

def main():
    print(f"[INFO] Using tokenizer: {model_tokenizer_name}")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_tokenizer_name)

    print(f"[INFO] Processing QA data from {input_json}...")
    filtered_data = filter_and_verify_squad(input_json)

    print("[INFO] Building tokenized examples...")
    examples = build_examples(filtered_data, tokenizer, max_length)
    print(f"[INFO] Total examples: {len(examples)}")

    print(f"[INFO] Saving to {output_pt}")
    torch.save(examples, output_pt)

    print("[DONE] Preprocessing completed.")

if __name__ == "__main__":
    main()


# 336 Questions are filtered out due to the impossibility factor/

# In[24]:


#!/usr/bin/env python
# evaluate_tydiqa_telugu.py

import os
import re
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    XLMRobertaForQuestionAnswering,
    XLMRobertaTokenizerFast
)

# ------------------
# Adjust paths here
# ------------------

DATA_PATH = "processed_indicqa/indicqa_windowed.pt"

MODEL_PATH = "../TeQAS 1.1/final_xlmr_tel_answerable_3_v2"  # Path to your fine-tuned QA model

print("\n[INFO] Loading processed dataset...")
data_list = torch.load(DATA_PATH)
dataset = Dataset.from_list(data_list)

print(f"[INFO] Loading model from {MODEL_PATH}...")
model = XLMRobertaForQuestionAnswering.from_pretrained(MODEL_PATH)
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def postprocess_qa_predictions(examples, start_logits, end_logits):
    """
    Convert model logits into text predictions by:
      - Finding best start/end
      - Using offset_mapping to slice the original context
    """
    preds = {}
    num_examples = len(examples)

    for i in range(num_examples):
        ex = examples[i]
        offsets = ex["offset_mapping"]
        context = ex["context"]

        # If mismatch in array sizes, skip
        if i >= len(start_logits) or i >= len(end_logits):
            preds[ex["id"]] = ""
            continue

        start_idx = int(np.argmax(start_logits[i]))
        end_idx   = int(np.argmax(end_logits[i]))

        # Check valid indices
        if (
            start_idx >= len(offsets) or
            end_idx   >= len(offsets) or
            start_idx > end_idx
        ):
            preds[ex["id"]] = ""
            continue

        start_char = offsets[start_idx][0]
        end_char   = offsets[end_idx][1]
        pred_text  = context[start_char:end_char]

        preds[ex["id"]] = pred_text

    return preds

def compute_metrics(eval_preds, examples):
    """
    Compute EM and F1 on the predictions vs. gold_text.
    """
    start_logits, end_logits = eval_preds

    # Convert any torch.Tensors to numpy
    if isinstance(start_logits, torch.Tensor):
        start_logits = start_logits.cpu().numpy()
    if isinstance(end_logits, torch.Tensor):
        end_logits = end_logits.cpu().numpy()

    predictions = postprocess_qa_predictions(examples, start_logits, end_logits)

    total_em, total_f1 = 0.0, 0.0
    for ex_idx, ex in enumerate(examples):
        ex_id = ex["id"]
        pred  = predictions.get(ex_id, "")
        gold  = ex["gold_text"]

        total_em += exact_match(pred, gold)
        total_f1 += f1_score(pred, gold)

    count = len(examples)
    return {
        "exact_match": 100.0 * total_em / count,
        "f1":          100.0 * total_f1 / count
    }

def exact_match(pred, gold):
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def f1_score(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    common   = set(pred_tokens) & set(gold_tokens)
    num_same = len(common)
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def normalize_text(s):
    """
    Lower text and remove punctuation, articles, extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def remove_punc(text):
        return re.sub(r"[^\w\s]", "", text)
    def white_space_fix(text):
        return " ".join(text.split())

    s = s.lower()
    s = remove_articles(s)
    s = remove_punc(s)
    s = white_space_fix(s)
    return s

print("\n[INFO] Running inference on all examples...")
start_logits_list = []
end_logits_list   = []

with torch.no_grad():
    for example in data_list:
        input_ids      = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits_list.append(outputs.start_logits.cpu().numpy())
        end_logits_list.append(outputs.end_logits.cpu().numpy())

# Concatenate to get final arrays
start_logits = np.concatenate(start_logits_list, axis=0)
end_logits   = np.concatenate(end_logits_list,   axis=0)

print("[INFO] Computing metrics...")
metrics = compute_metrics((start_logits, end_logits), data_list)

print("\n===== IndicQATe Final Evaluation Metrics =====")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")
print("====================================")


# ### Muril

# In[25]:


#!/usr/bin/env python
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# -------------------------
#  Adjust paths and params
# -------------------------
input_json = "indicqa_squad_window.json"  # Your windowed data
out_dir = "processed_indicqa_muril"
os.makedirs(out_dir, exist_ok=True)

output_pt = os.path.join(out_dir, "indicqa_windowed_muril.pt")

max_length = 512
model_tokenizer_name = "google/muril-large-cased"  # or "google/muril-base-cased"

def filter_and_verify_squad(input_path):
    """
    Loads and verifies answerable QAs from windowed data.
    """
    print(f"Loading data from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {
        "version": data.get("version", "windowed_indicqa"),
        "data": []
    }

    total_qas = 0
    valid_qas = 0
    
    print("Processing QA pairs...")
    for article in tqdm(data["data"]):
        new_paragraphs = []
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            new_qas = []
            
            for qa in paragraph["qas"]:
                total_qas += 1
                
                # Skip impossible questions
                if qa.get("is_impossible", True):
                    continue
                
                # Must have answers
                if not qa.get("answers"):
                    continue
                
                answer = qa["answers"][0]
                ans_text = answer.get("text")
                
                # Only check if answer exists in context
                if ans_text and ans_text in context:
                    new_qas.append(qa)
                    valid_qas += 1
            
            if new_qas:
                new_paragraphs.append({
                    "context": context,
                    "context_id": paragraph.get("context_id", ""),
                    "qas": new_qas
                })
        
        if new_paragraphs:
            new_data["data"].append({
                "title": article.get("title", ""),
                "paragraphs": new_paragraphs
            })
    
    print(f"\nTotal QAs: {total_qas}")
    print(f"Valid QAs: {valid_qas}")
    print(f"Filtered out: {total_qas - valid_qas}")
    
    return new_data

def build_examples(squad_data, tokenizer, max_length=512):
    """
    Build tokenized examples using MuRIL tokenizer.
    """
    examples_out = []
    print("Building tokenized examples...")
    
    for article in tqdm(squad_data["data"]):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_id = paragraph.get("context_id", "")
            
            for qa in paragraph["qas"]:
                ans = qa["answers"][0]
                ans_text = ans["text"]
                ans_start = ans.get("answer_start", context.find(ans_text))
                
                # If answer_start is invalid, find first occurrence
                if ans_start < 0 or ans_start >= len(context) or context[ans_start:ans_start + len(ans_text)] != ans_text:
                    ans_start = context.find(ans_text)
                
                # Skip if answer can't be found
                if ans_start == -1:
                    continue
                
                ans_end = ans_start + len(ans_text)
                
                # Tokenize
                encoding = tokenizer(
                    qa["question"],
                    context,
                    max_length=max_length,
                    truncation="only_second",
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    padding="max_length"
                )
                
                input_ids = encoding["input_ids"][0]
                attention_mask = encoding["attention_mask"][0]
                offset_mapping = encoding["offset_mapping"][0].tolist()
                
                # Find token indices for answer span
                start_token, end_token = None, None
                for i, (start_char, end_char) in enumerate(offset_mapping):
                    if start_char <= ans_start < end_char:
                        start_token = i
                    if start_char < ans_end <= end_char:
                        end_token = i
                
                # If exact mapping fails, use approximation
                if start_token is None or end_token is None or end_token < start_token:
                    # Find best approximate positions
                    for i, (start_char, end_char) in enumerate(offset_mapping):
                        if end_char > 0:  # Skip special tokens
                            start_token = i
                            break
                    
                    for i, (start_char, end_char) in enumerate(offset_mapping[start_token:], start_token):
                        if i >= len(offset_mapping) - 1 or offset_mapping[i+1][0] > ans_start + len(ans_text):
                            end_token = i
                            break
                    
                    if end_token is None:
                        end_token = start_token
                
                # Create example
                ex_item = {
                    "id": qa["id"],
                    "context_id": context_id,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "start_positions": torch.tensor(start_token, dtype=torch.long),
                    "end_positions": torch.tensor(end_token, dtype=torch.long),
                    "offset_mapping": offset_mapping,
                    "context": context,
                    "question": qa["question"],
                    "gold_text": ans_text,
                    "answer_start": ans_start
                }
                examples_out.append(ex_item)
    
    print(f"\nTotal examples built: {len(examples_out)}")
    return examples_out

def main():
    print(f"[INFO] Using tokenizer: {model_tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)

    print(f"[INFO] Processing QA data from {input_json}...")
    filtered_data = filter_and_verify_squad(input_json)

    print("[INFO] Building tokenized examples...")
    examples = build_examples(filtered_data, tokenizer, max_length)
    print(f"[INFO] Total examples: {len(examples)}")

    print(f"[INFO] Saving to {output_pt}")
    torch.save(examples, output_pt)
    print("[DONE] Preprocessing completed for MuRIL.")

if __name__ == "__main__":
    main()


# In[26]:
#!/usr/bin/env python
# evaluate_tydiqa_telugu_muril.py

import os
import re
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

############################################
# 1) Adjust paths and parameters
############################################
DATA_DIR   = "processed_indicqa_muril"
DATA_PATH  = os.path.join(DATA_DIR, "indicqa_windowed_muril.pt")
MODEL_PATH = "../TeQAS 1.2/final_muril_tel_answerable_v2"  # your fine-tuned MuRIL QA model folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the same MuRIL model name as in preprocessing
MURIL_TOKENIZER = "google/muril-large-cased"

############################################
# 2) Load Data & Model
############################################
print("[INFO] Loading processed dataset...")
examples_list = torch.load(DATA_PATH)
dataset = Dataset.from_list(examples_list)
print(f"[INFO] Loaded {len(dataset)} total examples for evaluation.")

print(f"[INFO] Loading fine-tuned model from {MODEL_PATH}...")
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MURIL_TOKENIZER)
model.to(device)
model.eval()

############################################
# 3) Define Postprocessing & Metrics
############################################
def normalize_text(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split())
    return s

def exact_match(pred, gold):
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def f1_score(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    common      = set(pred_tokens) & set(gold_tokens)
    num_same    = len(common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def postprocess_qa_predictions(examples, start_logits, end_logits):
    """Convert model logits into final text predictions."""
    predictions = {}
    for i, ex in enumerate(examples):
        if i >= len(start_logits) or i >= len(end_logits):
            predictions[ex["id"]] = ""
            continue

        offsets = ex["offset_mapping"]
        context = ex["context"]

        # Identify best start/end
        best_start = int(np.argmax(start_logits[i]))
        best_end   = int(np.argmax(end_logits[i]))

        # Validate token indices
        if best_start >= len(offsets) or best_end >= len(offsets) or best_start > best_end:
            predictions[ex["id"]] = ""
            continue

        # Convert token offsets to char positions, then slice context
        start_char = offsets[best_start][0]
        end_char   = offsets[best_end][1]
        pred_text  = context[start_char:end_char]

        predictions[ex["id"]] = pred_text
    return predictions

def compute_metrics(logits_tuple, examples):
    """
    logits_tuple => (start_logits, end_logits)
    examples => the raw examples with gold_text
    """
    start_logits, end_logits = logits_tuple

    # Convert to numpy if still Tensors
    if isinstance(start_logits, torch.Tensor):
        start_logits = start_logits.cpu().numpy()
    if isinstance(end_logits, torch.Tensor):
        end_logits = end_logits.cpu().numpy()

    # Postprocess predictions
    preds = postprocess_qa_predictions(examples, start_logits, end_logits)

    # Compute EM / F1
    total_em, total_f1, count = 0.0, 0.0, 0
    for ex in examples:
        gold = ex["gold_text"]
        pred = preds.get(ex["id"], "")
        total_em += exact_match(pred, gold)
        total_f1 += f1_score(pred, gold)
        count += 1

    em = 100.0 * total_em / count
    f1 = 100.0 * total_f1 / count
    return {"exact_match": em, "f1": f1}

############################################
# 4) Inference / Evaluation
############################################
print("[INFO] Running inference on each example...")
start_logits_list, end_logits_list = [], []

with torch.no_grad():
    for ex in examples_list:
        input_ids      = ex["input_ids"].unsqueeze(0).to(device)
        attention_mask = ex["attention_mask"].unsqueeze(0).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits_list.append(outputs.start_logits.cpu().numpy())
        end_logits_list.append(outputs.end_logits.cpu().numpy())

start_logits_all = np.concatenate(start_logits_list, axis=0)
end_logits_all   = np.concatenate(end_logits_list,   axis=0)

print("[INFO] Computing final metrics...")
metrics = compute_metrics((start_logits_all, end_logits_all), examples_list)

############################################
# 5) Print results
############################################
print("\n===== IndicQA (Telugu) - MuRIL Evaluation =====")
print(f"Exact Match (EM): {metrics['exact_match']:.2f}")
print(f"F1 Score:         {metrics['f1']:.2f}")
print("================================================\n")





