from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import json
import regex as re
import os
from PIL import Image
from typing import Optional
import pandas as pd
import glob
import io
from colpali_engine.utils.torch_utils import get_torch_device
from colpali_engine.models import ColQwen2, ColQwen2Processor
import json
from typing import Dict, List, Union
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S",)
logger = logging.getLogger(__name__)
from termcolor import colored

direct_answer = False
MODEL_PATH="model to test"
LORA_MODEL_NAME="Lora path"

DATA_ROOT = "../data/visa"

multi_image = False

setting = "multi" if multi_image else "single"
form = "direct" if direct_answer else "coe"

TEST_DATASETS = ['wiki']
OUTPUT_PATH=f"../output/rec_results_qwen2_5vl_{TEST_DATASETS[0]}_{setting}_{form}.jsonl"
os.makedirs("../output", exist_ok=True)


SYSTEM_PROMPT_DIRECT = """Given a document image and a relevant question, you should provide the final answer in the following format: "The answer is: ...".\n\n"""
SYSTEM_PROMPT_DIRECT_MULTI = """Given document images and a relevant question, you should provide the final answer in the following format: "The answer is: ...".\n\n"""

PROMPT = """[Task Description]: 
Given a document image and a relevant question, you should first analysis the image to extract information relevant to the question then provide the final answer. Finally please locate source to the final answer via a bounding box with image index.
[Restriction]:
1. For each identified element (e.g., figures, tables, or factual text) during analysis, provide a bounding box and include its image index to highlight the visual evidence.
2. The analysis and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis with visual evidence. </think>\n<answer> the final answer and the corresponding bounding box as its source. </answer>
3. Each bounding box must be formatted as:
Bounding box: {"bbox_2d": [x1, y1, x2, y2], "Image_index": 1}"""


PROMPT_MULTI = """[Task Description]: 
Given document images and a relevant question, you should first analysis the images to extract information relevant to the question then provide the final answer. Finally please locate source to the final answer via a bounding box with image index.
[Restriction]:
1. For each identified element (e.g., figures, tables, or factual text) during analysis, provide a bounding box and include its image index to highlight the visual evidence.
2. The analysis and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis with visual evidence. </think>\n<answer> the final answer and the corresponding bounding box as its source. </answer>
3. Each bounding box must be formatted as:
Bounding box: {"bbox_2d": [x1, y1, x2, y2], "Image_index": image_index_starting_from_1}"""

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
if LORA_MODEL_NAME:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_MODEL_NAME)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Post-process the generated text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\p{P}\p{S}]', '', text)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = text.strip()
    return text

def acc_compute(content, ground_truth):
    if direct_answer:
        answer_tag_pattern = r"^(.*?)The answer is: (.*)$"
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        student_answer = content_answer_match.group(2).strip() if content_answer_match else "unkonwn"
    else:
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        student_answer = content_answer_match.group(1).strip() if content_answer_match else "unknown"
    cleaned_content = re.sub(r"\nBounding box:\s*\{.*?\}(,\s*\{.*?\})*", "", student_answer)
    cleaned_content = cleaned_content.strip().strip(".")
    if normalize_text(ground_truth) in normalize_text(cleaned_content) or normalize_text(cleaned_content) in normalize_text(ground_truth):
        return 1
    else:
        return 0

def extract_metadata(input_str: str) -> Dict[str, Union[List[int], int]]:
    try:
        input_str = input_str.replace("'", '"')
        json_str_list = re.findall(r'\{.*?\}', input_str)
        if len(json_str_list) == 0:
            return [{
                "bbox": [0, 0, 0, 0],
                "image_id": 0
            }]
        data_all = []
        for json_str in json_str_list:
            data = json.loads(json_str)
            bbox_key = next((k for k in data if k.lower() == "bbox_2d"), None)
            index_key = next((k for k in data if k.lower() == "image_index"), None)
            bbox = data[bbox_key]
            image_id = data[index_key]
            data_all.append({
                "bbox": [int(x) for x in bbox],
                "image_id": int(image_id)
            })
        return data_all
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error: {str(e)}")
        return [{
            "bbox": [0, 0, 0, 0],
            "image_id": 0
        }]

def extract_bbox_answer(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if not content_answer_match:
        return None
    dic_list = extract_metadata(content_answer_match.group(1))
    return dic_list

def iou(box1, box2):
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return float(inter_area) / union_area

def has_similar_bbox(bboxes, box, iou_threshold=0.5):
    n = len(bboxes)
    if n > 0:
        for i in range(n):
            iou_value = iou(bboxes[i], box)
            if iou_value >= iou_threshold:
                return True
    return False

def score_multi_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs = [torch.nn.functional.normalize(q, p=2, dim=1) for q in qs]
        ps = [torch.nn.functional.normalize(p, p=2, dim=1) for p in ps]
        scores_list: List[torch.Tensor] = []

        for q, p in zip(qs, ps):
            q = q.to(device)  # shape: [n_tokens_q, embedding_dim]
            p = p.to(device)  # shape: [n_tokens_p, embedding_dim]

            cosine_similarity = torch.einsum("nd,md->nm", q, p)  # shape: [n_tokens_q, n_tokens_p]

            max_similarity = cosine_similarity.max(dim=1)[0]  # shape: [n_tokens_q]
            avg_similarity = max_similarity.mean()

            scores_list.append(avg_similarity.cpu())

        scores = torch.stack(scores_list)  # shape: [n_queries]
        return scores


def step_reward(content, candidiate, ids):
    think_tag_pattern = r'<think>(.*?)</think>'
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    source_pattern = r"Bounding box:\s*(\{.*?\})"
    reward = 0.0
    results = []
    results_iou = []
    has_similar_bbox_flag = False
    # Try symbolic verification first
    student_answer = "unkonw"
    try:
        content_think_match = re.search(think_tag_pattern, content, re.DOTALL)
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_think_match:
            think_content = content_think_match.group(1)
            student_answer = content_answer_match.group(1) if content_answer_match else "unkonw"
            previous_end = 0
            for match in re.finditer(source_pattern, think_content):
                image = None
                source_str = match.group(1)
                source_start = match.start()
                
                preceding_text = think_content[previous_end:source_start].strip()
                
                sentences = preceding_text.split(". ")
                last_sentence = sentences[-1].strip()
                previous_end = match.end()
                try:
                    source_dict = json.loads(source_str)
                    bbox_key = next((k for k in source_dict if k.lower() == "bbox_2d"), None)
                    index_key = next((k for k in source_dict if k.lower() == "image_index"), None)
                    bbox = source_dict[bbox_key]
                    image_id = source_dict[index_key]
                except json.JSONDecodeError:
                    bbox = None
                    image_id = None
                if bbox is not None:
                    try:
                        if len(candidiate) > 0:
                            image_name = candidiate[int(image_id) - 1]
                        else:
                            image_name = ids
                        image_tmp = Image.open(f"..data/visa/{TEST_DATASETS[0]}/image/{image_name}.bin")
                        weight = image_tmp.size[0]
                        height = image_tmp.size[1]
                        bbox = tuple(int(x) for x in bbox)

                        image = image_tmp.crop(bbox)
                        height_crop = image.size[1]
                        weight_crop = image.size[0]
                        if max(height_crop, weight_crop) / min(height_crop, weight_crop) > 200:
                            if height_crop > weight_crop:
                                image = image.resize((int(weight_crop * 200 / height_crop), 200))
                            else:
                                image = image.resize((200, int(height_crop * 200 / weight_crop)))
                    except:
                        image = None
                if image is not None and last_sentence != "" and (bbox[0] > 0 and bbox[1] > 0 and bbox[2] < weight and bbox[3] < height):
                    batch_images = [image]
                    batch_queries = [last_sentence]
                    batch_images = colqwen_processor.process_images(batch_images).to(process_model.device)
                    batch_queries = colqwen_processor.process_queries(batch_queries).to(process_model.device)
                    with torch.no_grad():
                        image_embeddings = process_model(**batch_images)
                        query_embeddings = process_model(**batch_queries)
                    scores =  score_multi_vector(query_embeddings, image_embeddings)
                    scores = scores[0].item()
                    results.append(scores)
                    if has_similar_bbox(results_iou, bbox):
                        has_similar_bbox_flag = True
                    results_iou.append(bbox)
                else:
                    results.append(0)
                    results_iou.append([0,0,0,0])
            if (min(results) if len(results) > 0 else 0) > 0.3:
                reward = 0.5
            if len(results_iou) > 1 and (not has_similar_bbox_flag) and reward == 0.5:
                reward += 0.5
            if int(idx) == -1 and multi_image and ("no answer" in normalize_text(student_answer)) and len(results) == 0:
                reward = 1.0
    except Exception as e:
        print(e)
    return reward

# Evaluate the `no answer` cases
def process_jsonl(input_file):
    total = 0
    correct = 0
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            if data.get('result', {}).get('pos_idx') == -1:
                total += 1
                if 'no answer' in normalize_text(data.get('result', {}).get('model_output', '')[0]):
                    correct += 1
    print(f"Total entries: {total}")
    print(f"Correct entries: {correct}")
    print(f"Accuracy: {correct / total if total > 0 else 0:.2f}")

if __name__ == '__main__':
    print("Start evaluation...")
    process_model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(model.device)
    process_model.eval()
    colqwen_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    for ds in TEST_DATASETS:
        print(f"Processing {ds}...")
        test_pattern = re.compile(r'^test-\d{5}-of-\d{5}\.parquet$')
        test_files = [f for f in glob.glob(f'{DATA_ROOT}/{ds}/data/*.parquet') if test_pattern.search(os.path.basename(f))]
        test_df = pd.concat((pd.read_parquet(f) for f in test_files), ignore_index=True)

        QUESTION_TEMPLATE = "{Question}"
        messages = []
        images= []
        ground_truths = []
        test_df = [row for index, row in test_df.iterrows() if len(row['candidates']) != 0]
        correct_number_box = 0
        correct_number_acc = 0
        correct_number_step = 0
        total_number = 0
        offset = 0
        with open(OUTPUT_PATH, "a") as f:
            for idx, row in enumerate(test_df[offset:]):
                row_dict = row.to_dict()
                if multi_image:
                    image = []
                    content = []
                    for i, path in enumerate(row_dict['candidates']):
                        content.append({"type": "text", "text": f"Document Image {i + 1}:\n"})
                        content.append({"type": "image"})
                        with open(f"../data/visa/{TEST_DATASETS[0]}/image/{path}.bin", "rb") as rb:
                            item = Image.open(io.BytesIO(rb.read()))
                            image.append(item)
                            content.append({"type": "text", "text": f"\tImageSize: {item.size}\n"})
                    content.append({"type": "text", "text": (SYSTEM_PROMPT_DIRECT_MULTI if direct_answer else PROMPT_MULTI) + "\n\nQuestion: "})
                    content.append({"type": "text", "text": row_dict["question"] + "?" if row_dict["question"][-1] != '?' else row_dict["question"]})
                    message = [{"role": "user", "content": content}]
                    index = row_dict['pos_idx'] + 1
                else:
                    image = Image.open(io.BytesIO(row_dict['image']['bytes']))
                    message = [
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Document Image: 1\n"},
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": "ImageSize: {}\n\n".format(image.size)},
                            {"type": "text", "text": (SYSTEM_PROMPT_DIRECT if direct_answer else PROMPT) + "\n\nQuestion: "},
                            {"type": "text", "text": row_dict["question"] + "?" if row_dict["question"][-1] != '?' else row_dict["question"]},
                        ]
                    }]
                    index = 1
                text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text,
                    images=image,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                batch_output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                correct = 0
                step_evalution = 0
                if not direct_answer:
                    model_answer = extract_bbox_answer(batch_output_text[0])
                    step_evalution = step_reward(batch_output_text[0], row_dict['candidates'], row_dict['id'])
                    ground_truth_box = row_dict['bounding_box'].tolist()
                    if model_answer is not None:
                        if any([iou(bbox_item['bbox'], ground_truth_box) > 0.5 and (bbox_item["image_id"] - 1 == row_dict['pos_idx'] if multi_image else True) for bbox_item in model_answer]):
                            correct = 1
                ground_truth_ans = row_dict['short_answer']
                if multi_image and row_dict['pos_idx'] == -1:
                    if ("no answer" in batch_output_text[0].lower()) or ("not" in batch_output_text[0].lower()):
                        correct_acc = 1
                        correct = 1 if not direct_answer else 0
                    else:
                        correct_acc = 0
                else:
                    correct_acc = acc_compute(batch_output_text[0], ground_truth_ans)
                correct_number_acc += correct_acc
                correct_number_box += correct
                correct_number_step += step_evalution
                total_number += 1
                result = {
                    'question': row_dict['question'],
                    'ground_truth': ground_truth_ans,
                    'model_output': batch_output_text,
                    "bounding_box": ground_truth_box,
                    'extracted_answer': model_answer,
                    'correct_number_box': correct_number_box / total_number,
                    'correct_number_acc': correct_number_acc / total_number,
                    'pos_idx': row_dict['pos_idx'],
                    'multi_image': multi_image,
                }
                logger.info(f"idx: {idx + offset}")
                logger.info(f"bbox: {ground_truth_box}")
                logger.info(f"output_text: {colored(batch_output_text, 'green')}")
                logger.info(f"ground_truth: {colored(ground_truth_ans, 'red')}")
                logger.info(f"correct_number_box: {correct_number_box / total_number}")
                logger.info(f"correct_number_acc: {correct_number_acc / total_number}")
                test_output = {
                    "idx": idx + offset,
                    "type": row_dict['short_answer_type'] if ds == 'fineweb' else row_dict['long_answer_type'],
                    "result": result
                }
                f.write(json.dumps(test_output) + "\n")


        # Calculate and print accuracy
        accuracy_box = correct_number_box / total_number * 100
        print(f"\nBox Accuracy of {ds}: {accuracy_box:.2f}%")
        accuracy = correct_number_acc / total_number * 100
        print(f"\nAcc Accuracy of {ds}: {accuracy:.2f}%")
        step_accuracy = correct_number_step / total_number * 100
        print(f"\nStep Accuracy of {ds}: {step_accuracy:.2f}%")
