import json
import logging
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SquadSpanValidator:
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def load_squad_data(self, dataset_path: str) -> Dict:
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            logger.error(f"Dataset file not found at {dataset_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in dataset file {dataset_path}")
            raise

    def validate_span(self, context: str, answer_text: str, start_position: int) -> bool:
        """Validate if answer text matches the context span"""
        if not answer_text:  
            return False
            
        end_position = start_position + len(answer_text)
        
        if start_position < 0 or end_position > len(context):
            return False
        
        try:
            span_text = context[start_position:end_position]
        except IndexError:
            logger.error(f"Invalid span: start={start_position}, end={end_position}, context_length={len(context)}")
            return False
        
        def normalize_text(text: str) -> str:
            return " ".join(text.strip().split()).lower()
        
        norm_answer = normalize_text(answer_text)
        norm_span = normalize_text(span_text)
        
        if norm_span == norm_answer:
            return True
        
        similarity = SequenceMatcher(None, norm_span, norm_answer).ratio()
        return similarity >= self.threshold

    def validate_dataset(self, dataset_path: str) -> Tuple[List, List, List, List]:
        """
        Validate answer spans for both answerable and impossible questions
        
        Returns:
            Tuple of (valid_answerable, invalid_answerable, valid_impossible, invalid_impossible) QA pairs
        """
        valid_answerable = []
        invalid_answerable = []
        valid_impossible = []
        invalid_impossible = []
        
        data = self.load_squad_data(dataset_path)
        
        for article in tqdm(data['data'], desc="Processing articles"):
            title = article.get('title', '')
            
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    qa_id = qa.get('id', '')
                    question = qa['question']
                    is_impossible = qa.get('is_impossible', False)
                    
                    qa_info = {
                        'id': qa_id,
                        'title': title,
                        'question': question,
                        'context': context,
                        'is_impossible': is_impossible
                    }
                    
                    if is_impossible:
                        # Validate plausible answers for impossible questions
                        if 'plausible_answers' in qa and qa['plausible_answers']:
                            plausible_answers = qa['plausible_answers']
                            qa_info['plausible_answers'] = plausible_answers
                            
                            # Check if any plausible answer span is valid
                            any_valid = False
                            for ans in plausible_answers:
                                answer_text = ans['text']
                                start_position = ans['answer_start']
                                
                                if self.validate_span(context, answer_text, start_position):
                                    any_valid = True
                                    break
                            
                            if any_valid:
                                valid_impossible.append(qa_info)
                            else:
                                invalid_impossible.append(qa_info)
                        else:
                            # No plausible answers provided
                            invalid_impossible.append(qa_info)
                            
                    else:
                        # Validate answerable questions
                        if 'answers' in qa and qa['answers']:
                            answers = qa['answers']
                            qa_info['answers'] = answers
                            
                            # Check if any answer span is valid
                            any_valid = False
                            for ans in answers:
                                answer_text = ans['text']
                                start_position = ans['answer_start']
                                
                                if self.validate_span(context, answer_text, start_position):
                                    any_valid = True
                                    break
                            
                            if any_valid:
                                valid_answerable.append(qa_info)
                            else:
                                invalid_answerable.append(qa_info)
                        else:
                            invalid_answerable.append(qa_info)
        
        return valid_answerable, invalid_answerable, valid_impossible, invalid_impossible

    def analyze_results(self, valid_ans: List, invalid_ans: List, valid_imp: List, invalid_imp: List):
        """Print analysis of validation results"""
        total = len(valid_ans) + len(invalid_ans) + len(valid_imp) + len(invalid_imp)
        
        print("\nValidation Results:")
        print(f"Total QA pairs: {total}")
        
        print("\nAnswerable Questions:")
        print(f"Valid spans: {len(valid_ans)} ({(len(valid_ans)/total)*100:.2f}%)")
        print(f"Invalid spans: {len(invalid_ans)} ({(len(invalid_ans)/total)*100:.2f}%)")
        
        print("\nImpossible Questions:")
        print(f"Valid plausible spans: {len(valid_imp)} ({(len(valid_imp)/total)*100:.2f}%)")
        print(f"Invalid plausible spans: {len(invalid_imp)} ({(len(invalid_imp)/total)*100:.2f}%)")

        # Print some example invalid spans
        if invalid_ans or invalid_imp:
            print("\nExample Invalid Spans:")
            
            if invalid_ans:
                print("\nAnswerable Questions:")
                for i, qa in enumerate(invalid_ans[:2]):
                    print(f"\nExample {i+1}:")
                    print(f"Question: {qa['question']}")
                    for ans in qa['answers']:
                        print(f"Answer: {ans['text']}")
                        start = ans['answer_start']
                        print(f"Context Span: {qa['context'][start:start+len(ans['text'])]}")
                    
            if invalid_imp:
                print("\nImpossible Questions:")
                for i, qa in enumerate(invalid_imp[:2]):
                    print(f"\nExample {i+1}:")
                    print(f"Question: {qa['question']}")
                    if 'plausible_answers' in qa:
                        for ans in qa['plausible_answers']:
                            print(f"Plausible Answer: {ans['text']}")
                            start = ans['answer_start']
                            print(f"Context Span: {qa['context'][start:start+len(ans['text'])]}")

def validate_squad_splits(train_path: str, val_path: str, test_path: str):
    """Validate all dataset splits"""
    validator = SquadSpanValidator()
    
    for split_name, path in [("Training", train_path), 
                           ("Validation", val_path),
                           ("Test", test_path)]:
        print(f"\nProcessing {split_name} Split:")
        try:
            valid_ans, invalid_ans, valid_imp, invalid_imp = validator.validate_dataset(path)
            validator.analyze_results(valid_ans, invalid_ans, valid_imp, invalid_imp)
            
        except Exception as e:
            logger.error(f"Error processing {split_name} split: {str(e)}")

if __name__ == "__main__":
    # Dataset paths
    # train_json = "Telugu Data/squad2.0_telugu_train.json"
    # val_json = "Telugu Data/squad2.0_telugu_val.json" 
    # test_json = "Telugu Data/squad2.0_telugu_test.json"

    train_json = "English Data/squad2.0_train.json"
    val_json = "English Data/squad2.0_val.json"
    test_json = "English Data/squad2.0_test.json"

    # Validate all splits
    validate_squad_splits(train_json, val_json, test_json)

# import json
# import logging
# import csv
# from typing import Dict, List, Tuple
# from difflib import SequenceMatcher
# from tqdm import tqdm

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SquadSpanValidator:
#     def __init__(self, threshold: float = 0.99, suffixes: List[str] = None):
#         """
#         Initialize the validator with a similarity threshold and optional suffix handling.
#         """
#         self.threshold = threshold
#         self.suffixes = suffixes or ['లో', 'కు', 'తో', 'యందు']  # Common Telugu suffixes

#     def load_squad_data(self, dataset_path: str) -> Dict:
#         """
#         Load the SQuAD-style dataset from a JSON file.
#         """
#         try:
#             with open(dataset_path, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         except (FileNotFoundError, json.JSONDecodeError) as e:
#             logger.error(f"Error loading dataset: {e}")
#             raise

#     def normalize_text(self, text: str) -> str:
#         """
#         Normalize text by stripping whitespace and converting to lowercase.
#         """
#         return " ".join(text.strip().split()).lower()

#     def strip_suffixes(self, text: str) -> str:
#         """
#         Remove common Telugu suffixes from the text.
#         """
#         for suffix in self.suffixes:
#             if text.endswith(suffix):
#                 return text[: -len(suffix)]
#         return text

#     def validate_span(self, context: str, answer_text: str, start_position: int) -> Tuple[bool, float, str]:
#         """
#         Validate if the answer_text matches the span in the context starting at start_position.
#         Returns a tuple of (is_valid, similarity_score, extracted_span).
#         """
#         if not answer_text or start_position < 0 or start_position >= len(context):
#             return False, 0.0, ""

#         # Extract context span
#         end_position = start_position + len(answer_text)
#         if end_position > len(context):
#             return False, 0.0, ""

#         span_text = context[start_position:end_position]
#         norm_span = self.strip_suffixes(self.normalize_text(span_text))
#         norm_answer = self.strip_suffixes(self.normalize_text(answer_text))

#         # Exact match
#         if norm_span == norm_answer:
#             return True, 1.0, span_text

#         # Similarity check
#         similarity = SequenceMatcher(None, norm_span, norm_answer).ratio()
#         return similarity >= self.threshold, similarity, span_text

#     def validate_dataset(self, dataset_path: str) -> Tuple[List, List, List, List]:
#         """
#         Validate all QA pairs in the dataset and categorize them into valid/invalid lists.
#         """
#         valid_answerable = []
#         invalid_answerable = []
#         valid_impossible = []
#         invalid_impossible = []

#         data = self.load_squad_data(dataset_path)

#         for article in tqdm(data['data'], desc="Processing articles"):
#             for paragraph in article['paragraphs']:
#                 context = paragraph['context']

#                 for qa in paragraph['qas']:
#                     qa_id = qa.get('id', '')
#                     question = qa['question']
#                     is_impossible = qa.get('is_impossible', False)

#                     qa_info = {
#                         'id': qa_id,
#                         'question': question,
#                         'context': context,
#                         'is_impossible': is_impossible
#                     }

#                     if is_impossible:
#                         plausible_answers = qa.get('plausible_answers', [])
#                         any_valid = False
#                         for ans in plausible_answers:
#                             answer_text = ans['text']
#                             start_position = ans['answer_start']
#                             is_valid, similarity, span_text = self.validate_span(context, answer_text, start_position)

#                             if is_valid:
#                                 any_valid = True
#                             else:
#                                 qa_info.update({
#                                     'answer_text': answer_text,
#                                     'start_position': start_position,
#                                     'span_text': span_text,
#                                     'similarity': similarity
#                                 })
#                                 invalid_impossible.append(qa_info)
#                         if any_valid:
#                             valid_impossible.append(qa_info)
#                     else:
#                         answers = qa.get('answers', [])
#                         any_valid = False
#                         for ans in answers:
#                             answer_text = ans['text']
#                             start_position = ans['answer_start']
#                             is_valid, similarity, span_text = self.validate_span(context, answer_text, start_position)

#                             if is_valid:
#                                 any_valid = True
#                             else:
#                                 qa_info.update({
#                                     'answer_text': answer_text,
#                                     'start_position': start_position,
#                                     'span_text': span_text,
#                                     'similarity': similarity
#                                 })
#                                 invalid_answerable.append(qa_info)
#                         if any_valid:
#                             valid_answerable.append(qa_info)

#         return valid_answerable, invalid_answerable, valid_impossible, invalid_impossible

#     def save_invalid_to_csv(self, invalid_ans: List, invalid_imp: List, output_path: str):
#         """
#         Save invalid spans to a CSV file for analysis.
#         """
#         fieldnames = ['id', 'question', 'answer_text', 'start_position', 'span_text', 'similarity', 'context']
#         with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             for qa in invalid_ans + invalid_imp:
#                 writer.writerow({
#                     'id': qa['id'],
#                     'question': qa['question'],
#                     'answer_text': qa.get('answer_text', ''),
#                     'start_position': qa.get('start_position', ''),
#                     'span_text': qa.get('span_text', ''),
#                     'similarity': qa.get('similarity', ''),
#                     'context': qa['context']
#                 })

# def validate_squad_splits(train_path: str, val_path: str, test_path: str, output_csv: str):
#     """
#     Validate SQuAD-style dataset splits and save invalid spans to a CSV file.
#     """
#     validator = SquadSpanValidator(threshold=0.99)

#     for split_name, path in [("Training", train_path), ("Validation", val_path), ("Test", test_path)]:
#         print(f"\nProcessing {split_name} Split:")
#         valid_ans, invalid_ans, valid_imp, invalid_imp = validator.validate_dataset(path)
#         print(f"{split_name} Results:")
#         print(f"Invalid Answerable: {len(invalid_ans)}")
#         print(f"Invalid Impossible: {len(invalid_imp)}")
#         validator.save_invalid_to_csv(invalid_ans, invalid_imp, f"{output_csv}_{split_name.lower()}.csv")

# if __name__ == "__main__":
#     # Dataset paths
#     train_json = "squad2.0_telugu_train.json"
#     val_json = "squad2.0_telugu_val.json"
#     test_json = "squad2.0_telugu_test.json"

#     # train_json = "English Data/squad2.0_train.json"
#     # val_json = "English Data/squad2.0_val.json"
#     # test_json = "English Data/squad2.0_test.json"

#     output_csv = "invalid_spans"

#     # Validate all splits and save invalid spans
#     validate_squad_splits(train_json, val_json, test_json, output_csv)