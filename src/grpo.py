import io
import json
import torch
import regex as re
import polars as pl
from PIL import Image
from collections import Counter
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from swanlab.integration.transformers import SwanLabCallback
from utils.grpo_trainer import Qwen2VLGRPOTrainer
from utils.grpo_config import GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from typing import Tuple, List, Optional, Union
from colpali_engine.utils.torch_utils import get_torch_device
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward
# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "accuracy", "box", "step"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    single_image: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use single image or multi-image"},
    )
    dataset_name: Optional[str] = field(
        default="wiki",
        metadata={"help": "The name of the dataset to use."},
    )
    all_data: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use all data or not."},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

def score_multi_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
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

# ----------------------- Prompt -----------------------
SYSTEM_PROMPT_CORRECT_SINGLE = """[Task Description]: 
Given a document image and a relevant question, you should first analysis the image to extract information relevant to the question then provide the final answer. Finally please locate source to the final answer via a bounding box with image index.
[Restriction]:
1. For each identified element (e.g., figures, tables, or factual text) during analysis, provide a bounding box and include its image index to highlight the visual evidence.
2. The analysis and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis with visual evidence. </think>\n<answer> the final answer and the corresponding bounding box as its source. </answer>
3. Each bounding box must be formatted as:
Bounding box: {"bbox_2d": [x1, y1, x2, y2], "Image_index": 1}"""

SYSTEM_PROMPT_CORRECT_MULTI = """[Task Description]: 
Given document images and a relevant question, you should first analysis the images to extract information relevant to the question then provide the final answer. Finally please locate source to the final answer via a bounding box with image index.
[Restriction]:
1. For each identified element (e.g., figures, tables, or factual text) during analysis, provide a bounding box and include its image index to highlight the visual evidence.
2. The analysis and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis with visual evidence. </think>\n<answer> the final answer and the corresponding bounding box as its source. </answer>
3. Each bounding box must be formatted as:
Bounding box: {"bbox_2d": [x1, y1, x2, y2], "Image_index": image_index_starting_from_1}"""


def calculate_f1_reward(pred_norm: str, gt_norm: str) -> float:
    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()
    
    common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common_tokens.values())
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    recall = num_common / (len(gt_tokens))
        
    return recall


columns_to_read = ['id', 'question', 'short_answer', 'bounding_box', 'candidates', 'pos_idx']
class LazySupervisedDataset(Dataset):
    def __init__(self, data_name: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.ds = script_args.dataset_name
        if 'visa' in data_name:
            df_all = pl.read_parquet(f"../data/rl/train.parquet")
            if SINGLE_IMAGE:
                self.list_data_dict = [row for idx, row in enumerate(df_all.iter_rows(named=True))]
            else:
                df_all = df_all.sample(n=df_all.height, shuffle=True, seed=42)
                self.list_data_dict = [row for idx, row in enumerate(df_all.iter_rows(named=True)) if (len(row['candidates']) > 0)]

            if not script_args.all_data:
                self.list_data_dict = [row for row in self.list_data_dict if self.ds in row['file']]

            print(len(self.list_data_dict))
            
        else:
            ValueError("Invalid dataset name")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        def make_conversation_image(example, image_size):
            return  [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Document Image 1:\n"},
                        {"type": "image"},
                        {"type": "text", "text": "ImageSize: {}\n\n".format(image_size)},
                        {"type": "text", "text": SYSTEM_PROMPT_CORRECT + "\n\nQuestion: "},   
                        {"type": "text", "text": example["question"] + "?" if example["question"][-1] != '?' else example["question"]},
                    ],
                }
            ]

        example = self.list_data_dict[i]
        if SINGLE_IMAGE:
            example['candidates'] = []
        if len(example['candidates']) > 0:
            image = []
            content = []
            for i, path in enumerate(example['candidates']):
                content.append({"type": "text", "text": f"Document Image {i + 1}:\n"})
                content.append({"type": "image"})
                with open(f"./data/{example['file']}/image/{path}.bin", "rb") as rb:
                    item = Image.open(io.BytesIO(rb.read()))
                    image.append(item)
                    content.append({"type": "text", "text": f"ImageSize: {item.size}\n\n"})
            content.append({"type": "text", "text": SYSTEM_PROMPT_CORRECT + "\n\nQuestion: "})
            content.append({"type": "text", "text": example["question"] + "?" if example["question"][-1] != '?' else example["question"]})
            example["messages"] = [
                {"role": "user", "content": content},
            ]
        else:
            path = example['id']
            with open(f"../data/visa/{example['file']}/image/{path}.bin", "rb") as rb:
                image = Image.open(io.BytesIO(rb.read()))
            example["messages"] = make_conversation_image(example, image.size)
        return {
            'id': example['id'],
            'image': image,
            'problem': example['question'],
            'candidates': example['candidates'],
            'solution': example['bounding_box'],
            'prompt': example["messages"],
            'index_page': example['pos_idx'] if len(example['candidates']) > 0 else 0,
            'answer': example['short_answer'],
            "file": example['file'],
        }

'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''
def iou(box1, box2):
    if len(box1) != 4:
        return 0
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1)*(inter_y2-inter_y1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

def iou_reward(completions, solution, **kwargs):
    """
    completions = [[{"role": "assistant", "content": completion}] for completion in completions]
    """
    contents = [completion[0]["content"] for completion in completions]

    index_page = kwargs.get("index_page", None)
    rewards = []
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r"<think>(.*?)</think>"
    for content, index, sol in zip(contents, index_page, solution):
        reward = 0.0
        try:
            content_match = re.search(answer_tag_pattern, content, re.DOTALL)
            think_content_match = re.search(think_tag_pattern, content, re.DOTALL)
            if content_match and think_content_match:
                content_answer_match = content_match.group(1)
                think_match = think_content_match.group(1)
            else:
                content_answer_match = "Error"
                think_match = "Error"
            content = content_answer_match.replace("'", '"')
            json_str_list = re.findall(r'\{.*?\}', content)
            if len(json_str_list) != 0:
                data_all = []
                all_right = True 
                for json_str in json_str_list:
                    data = json.loads(json_str)
                    bbox_key = next((k for k in data if k.lower() == "bbox_2d"), None)
                    index_key = next((k for k in data if k.lower() == "image_index"), None)
                    bbox = data[bbox_key]
                    image_id = data[index_key]
                    data_all.append({"bbox": [int(x) for x in bbox], "image_index": int(image_id)})
                    if json_str not in think_match:
                        all_right = False
                iou_2_list = [(iou(bbox_item['bbox'], sol) > 0.5) and (bbox_item["image_index"] - 1) == int(index) for bbox_item in data_all]
                if any(iou_2_list) and all_right:
                    reward = 1.0
                if int(index) == -1 and ("no answer" in normalize_text(content_answer_match) or "not" in normalize_text(content_answer_match)):
                    reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content.strip(), re.DOTALL) for content in completion_contents]

    return [1.0 if match else -1.0 for match in matches]  # TODO

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\p{P}\p{S}]', '', text)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = text.strip()
    return text

def accuracy_reward(completions, **kwargs):
    # print(completions)
    print(completions[0][0]["content"])
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    answers = kwargs.get("answer", None)
    index = kwargs.get("index_page", None)
    for content, sol, idx in zip(contents, answers, index):
        reward = 0.0
        
        content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else "unkonw"
        cleaned_content = re.sub(r"Bounding box:\s*\{.*?\}(,\s*\{.*?\})*", "", student_answer)
        cleaned_content = cleaned_content.strip().strip(".")

        if (normalize_text(sol) in normalize_text(cleaned_content) or normalize_text(cleaned_content) in normalize_text(sol)): 
            reward = 0.5
        
        reward += calculate_f1_reward(normalize_text(cleaned_content), normalize_text(sol)) * 0.5
        if int(idx) == -1:
            if "no answer" in normalize_text(student_answer) or "not" in normalize_text(student_answer):
                reward = 1.0
            else:
                reward = 0.0

                
        rewards.append(reward)
    return rewards

def has_similar_bbox(bboxes, result_id, box, image_id, iou_threshold=0.5):
    n = len(bboxes)
    if n > 0:
        for i in range(n):
            iou_value = iou(bboxes[i], box)
            if iou_value >= iou_threshold and result_id[i] == image_id:
                return True
    return False

def step_reward(completions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    think_tag_pattern = r"<think>(.*?)</think>"
    source_pattern = r"Bounding box:\s*(\{.*?\})"
    candidiate_ids = kwargs.get("candidates", None)
    file_name = kwargs.get("file", None)
    image_id = kwargs.get("id", None)
    index_page = kwargs.get("index_page", None)
    answers = kwargs.get("answer", None)
    for content, candidiate, ids, sol, idx, ds in zip(contents, candidiate_ids, image_id, answers, index_page, file_name):
        reward = 0.0
        reward_acc = 0
        results = []
        results_iou = []
        result_id = []
        has_similar_bbox_flag = False
        student_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        # Try symbolic verification first
        student_answer = student_match.group(1) if student_match else "unknown"
        try:
            content_answer_match = re.search(think_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                think_content = content_answer_match.group(1)
                previous_end = 0
                for match in re.finditer(source_pattern, think_content):
                    image = None
                    source_str = match.group(1)
                    source_start = match.start()
                    
                    preceding_text = think_content[previous_end:source_start].strip()
                    
                    sentences = re.split(r'\. |\n', preceding_text)
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
                            image_tmp = Image.open(f"../data/visa/{ds}/image/{image_name}.bin")
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
                        if has_similar_bbox(results_iou, result_id, bbox, image_id):
                            has_similar_bbox_flag = True
                        results_iou.append(bbox)
                        result_id.append(image_id)
                    else:
                        results.append(0)
                        results_iou.append([0,0,0,0])
                        result_id.append(0)
                if (min(results) if len(results) > 0 else 0) > 0.3 and len(results_iou) > 1:
                    reward = 0.5
                if not has_similar_bbox_flag and reward == 0.5:
                    reward += 0.5
                cleaned_content = re.sub(r"Bounding box:\s*\{.*?\}(,\s*\{.*?\})*", "", student_answer)
                cleaned_content = cleaned_content.strip().strip(".")

                if (normalize_text(sol) in normalize_text(cleaned_content) or normalize_text(cleaned_content) in normalize_text(sol)): 
                    reward_acc = 0.5

                reward_acc += calculate_f1_reward(normalize_text(cleaned_content), normalize_text(sol)) * 0.5

                if reward_acc < 0.4:
                    reward = 0
                
                if int(idx) == -1:
                    reward = 0.0
                if int(idx) == -1 and ("no answer" in normalize_text(student_answer) or "not" in normalize_text(student_answer)):
                    reward = 1.0

                
        except Exception as e:
            print(e)
        rewards.append(reward)
    return rewards


reward_funcs_registry = {
    "box": iou_reward,
    "format": format_reward,
    "accuracy": accuracy_reward,
    "step": step_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    swanlab_callback = SwanLabCallback(
        project="grpo_qwen2.5", 
        experiment_name="TransformersTest"
    )
    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
        callbacks=[swanlab_callback],
    )

    # Train and push the model to the Hub
    trainer.train(resume_from_checkpoint=False)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    from accelerate import Accelerator
    accelerator = Accelerator()
    process_model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).to(accelerator.device)
    process_model.eval()
    colqwen_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    script_args, training_args, model_args = parser.parse_args_and_config()
    SINGLE_IMAGE = True if script_args.single_image else False
    SYSTEM_PROMPT_CORRECT = SYSTEM_PROMPT_CORRECT_SINGLE if SINGLE_IMAGE else SYSTEM_PROMPT_CORRECT_MULTI
    main(script_args, training_args, model_args)

