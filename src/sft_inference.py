import logging
import os
import sys
import datasets
import torch
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from utils.configs import SFTConfig
from utils.callbacks import get_callbacks
import json
from PIL import Image
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import field
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from typing import Optional
import glob
import io
import polars as pl
import torch
from typing import Tuple
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
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

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})
    meta_learning: bool = field(default=False, metadata={"help": "Whether to use meta learning."})
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    multi_image: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use multiple images"},
    )
    freeze_vision_modules: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze vision modules"},
    )
    lora_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the LoRA model."},  
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

DATA_ROOT = "./data/cold_start_input"
processor = None
vision_modules_keywords = ["visual"]
columns_to_read = ['image', 'id', 'question', 'short_answer', 'bounding_box', 'candidates']
        
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


class LazySupervisedDataset(Dataset):
    def __init__(self, data_name: str, script_args: ScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.ds = script_args.dataset_name
        if 'visa' in data_name:
            train_dataset = glob.glob(os.path.join(DATA_ROOT, "*.parquet"))
            for file in train_dataset:
                df = pl.read_parquet(file).select(columns_to_read)
                file_name = "../data/r1_template/modified_{}.json".format(file.split("/")[-1])

                with open(file_name, "r") as r:
                    json_data = json.load(r)
                    json_df = pl.DataFrame(json_data)
                    df = df.head(len(json_data))
                    df = df.with_columns(json_df['result'])
                    df = df.with_columns(json_df['save'])
                    df = df.with_columns(json_df['file'])
                    df = df.with_columns(json_df['candidate'])
                    print(f"Processing file: {file}, len(df): {len(df)}")
                    if 'train_df' in locals():
                        train_df = pl.concat([train_df,df])
                    else:
                        train_df = df
            train_df = train_df.filter(pl.col('save') == True)

            train_df = train_df.sample(fraction=1.0, shuffle=True, seed=3407)

            if SINGLE_IMAGE:
                self.list_data_dict = [row for row in train_df.iter_rows(named=True) if (row["candidate"] == '[]')]
            else:
                self.list_data_dict = [row for row in train_df.iter_rows(named=True) if (row["candidate"] != '[]')] 
            print(len(self.list_data_dict))

            if not script_args.all_data:
                self.list_data_dict = [row for row in self.list_data_dict if self.ds in row['file']]

            
            print()
        else:
            ValueError("Invalid dataset name")

    def __len__(self):
        logger.info(f"Length of dataset: {len(self.list_data_dict)}")
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
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
                },
                {
                    "role": "assistant",
                    "content": example['result']
                }
            ]

        example = self.list_data_dict[i]
        if SINGLE_IMAGE:
            example['candidates'] = []
        if len(example['candidates']) > 0:
            if 'paper' in example['file']:
                ds = 'paper'
            elif 'wiki' in example['file']:
                ds = 'wiki'
            else:
                ds = 'fineweb'
            image_for_train = []
            content = []
            for i, path in enumerate(example['candidates']):
                content.append({"type": "text", "text": f"Document Image {i + 1}:\n"})
                content.append({"type": "image"})
                with open(f"../data/visa/{ds}/image/{path}.bin", "rb") as rb:
                    item = Image.open(io.BytesIO(rb.read()))
                    image_for_train.append(item)
                    content.append({"type": "text", "text": f"ImageSize: {item.size}\n\n"})
            content.append({"type": "text", "text": SYSTEM_PROMPT_CORRECT + "\n\nQuestion: "})
            content.append({"type": "text", "text": example["question"] + "?" if example["question"][-1] != '?' else example["question"]})
            example["messages"] = [
                {"role": "user", "content": content},
                {
                    "role": "assistant",
                    "content": example["result"]
                }
            ]
        else:
            image_sigle = Image.open(io.BytesIO(example['image']['bytes']))
            image_for_train = image_sigle
            example["messages"] = make_conversation_image(example, image_sigle.size)
        example['image_open'] = image_for_train
        return example

def collate_fn(examples):
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False).strip()
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        image_inputs.append(example['image_open'])
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding_side="left",
        padding=True,
    )
    res = [example["messages"][1]['content'] for example in examples]
    batch_answer = [
        processor.tokenizer(f"{example}", add_special_tokens=False)
        for example in res
    ]
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for i, label in enumerate(batch_answer):
        valid_length = len(label["input_ids"])
        labels[i, :- valid_length - 1] = -100
    batch["labels"] = labels
    

    return batch

def find_all_linear_names(model, multimodal_keywords):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        # LoRA is not applied to the vision modules
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            if isinstance(module, cls) and "patch_embed" not in name and ('proj' in name or 'qkv' in name) and SINGLE_IMAGE:
                lora_module_names.add(name)
            else:
                continue
        if isinstance(module, cls) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name or 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name):
            lora_module_names.add(name)
    for m in lora_module_names:
        if "embed_tokens" in m:
            lora_module_names.remove(m)
    return list(lora_module_names)

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    script_args.per_device_train_batch_size = training_args.per_device_train_batch_size
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    model_path = model_args.model_name_or_path if SINGLE_IMAGE else model_args.lora_name_or_path
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    target_modules = find_all_linear_names(model, ['visual'], SINGLE_IMAGE)
    print(f"Applying PEFT to the following modules: {target_modules}")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM
    )
    lora_config.target_modules = target_modules # target_modules
    model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    from swanlab.integration.transformers import SwanLabCallback
    swanlab_callback = SwanLabCallback(
        project="sft_qwen2.5", 
        experiment_name="TransformersTest_sft"
    )
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        callbacks=[swanlab_callback],
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = False
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)




if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    SINGLE_IMAGE = True if script_args.single_image else False
    SYSTEM_PROMPT_CORRECT = SYSTEM_PROMPT_CORRECT_SINGLE if SINGLE_IMAGE else SYSTEM_PROMPT_CORRECT_MULTI
    main(script_args, training_args, model_args)
