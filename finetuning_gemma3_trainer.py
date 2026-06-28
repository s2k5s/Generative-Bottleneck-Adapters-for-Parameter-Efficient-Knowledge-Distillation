import sys
import torch
import datasets
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field # ### GEMMA: Added for KD arguments

from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed,
    DataCollatorForLanguageModeling # ### GEMMA: This might be a better default
)
from transformers.trainer_utils import EvaluationStrategy

### GEMMA: Import your custom adapter-Gemma model and its config
# (Update 'gemma_adapter_model' to your actual file name)
from third_party.models.modeling_gemma import Gemma3TextConfig, Gemma3ForConditionalGeneration

### GEMMA: Import the *standard* Gemma model for the teacher
from transformers import Gemma3ForConditionalGeneration as StandardGemmaForConditionalGeneration

### GEMMA: Import the new KD Trainer
# (Update 'gemma_kd_trainer' to your actual file name)
from third_party.trainers.gemma3_trainer import GemmaAdapterKDTrainer

from adapters import AdapterController, AutoAdapterConfig
from data import AutoTask

### WARNING: This collator MUST be updated for Gemma (CausalLM)
# It needs to create 'labels' from 'input_ids' and use -100 for masking
# in addition to adding the 'task' field.
from third_party.utils import TaskCollator_gemma, check_output_dir

from metrics import build_compute_metrics_fn
from training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, \
    AdapterTrainingArguments, KDArguments
from utils import freezing_params, get_last_checkpoint_path, create_dir,\
    handle_metrics, get_training_args

logger = logging.getLogger("log.txt")

import dataclasses  # <-- Add this import
from adapters import MetaAdapterConfig  # <-- Import your config class (adjust path if needed)
from transformers import GenerationConfig # <-- Make sure this is imported

# --- MONKEY-PATCH ---
# This dynamically adds the 'to_dict' method that transformers is looking for.
def meta_adapter_config_to_dict(self):
    return dataclasses.asdict(self)

MetaAdapterConfig.to_dict = meta_adapter_config_to_dict
def remove_rank_info_from_argv(args):
    extra_parameters = {}
    if args[1].startswith("--local_rank"):
        extra_parameters.update({'local_rank': int(args[1].split('=')[-1])})
        del args[1]
    return extra_parameters

def main():
    # See all possible arguments in src/transformers/training_args.py
    # We now add KDArguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterTrainingArguments, KDArguments)) # ### KD: Added KDArguments

    # --- Argument parsing logic (unchanged, but now parses 5 groups) ---
    if len(sys.argv) > 2 and sys.argv[1].startswith("--local_rank") and (sys.argv[2].endswith(".json")):
        rank_info = remove_rank_info_from_argv(sys.argv)
        args_dict = json.loads(Path(sys.argv[1]).read_text())
        args_dict.update(rank_info)
        model_args, data_args, training_args, adapter_args, kd_args = parser.parse_dict(args_dict) # ### KD: Added kd_args
        print(training_args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        logger.warning("config path: %s", sys.argv[1])
        model_args, data_args, training_args, adapter_args, kd_args = parser.parse_json_file( # ### KD: Added kd_args
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args, kd_args = parser.parse_args_into_dataclasses() # ### KD: Added kd_args
    check_output_dir(training_args)
    # --- End argument parsing ---
    # Setup logging (unchanged)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed (unchanged)
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    print(model_args.config_name if model_args.config_name else model_args.model_name_or_path)
    ### GEMMA: Load Gemma config instead of T5
    config = Gemma3TextConfig.from_pretrained(
        model_args.config_name if model_args.config_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    print(f"1b Config : {config}")
    
    # This logic for updating config remains valid for Gemma
    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout",
                          "attention_dropout",  "train_adapters")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            if hasattr(config, p):
                setattr(config, p, getattr(training_args, p))
            else:
                logger.warning(f"Config ({config.__class__.__name__}) doesn't have a `{p}` attribute")

    # Adapter config logic (unchanged)
    if training_args.train_adapters:
        adapter_config = AutoAdapterConfig.get(adapter_args.adapter_config_name)
        adapter_config.input_dim = config.hidden_size
        adapter_config.tasks = data_args.tasks
        adapter_config.task_to_adapter = {task:adapter for task, adapter in zip(data_args.tasks, data_args.adapters)} if data_args.adapters is not None else None
        adapter_config.task_to_embeddings = {task:embedding for task, embedding in zip(data_args.tasks, data_args.task_embeddings)}\
            if (data_args.task_embeddings is not None) else None
        extra_adapter_params = ("task_embedding_dim",
                                "add_layer_norm_before_adapter",
                                "add_layer_norm_after_adapter",
                                "reduction_factor",
                                "hidden_dim",
                                "non_linearity",
                                "train_task_embeddings",
                                "projected_task_embedding_dim",
                                "task_hidden_dim",
                                "conditional_layer_norm",
                                "train_adapters_blocks",
                                "unique_hyper_net",
                                "unique_hyper_net_layer_norm",
                                "efficient_unique_hyper_net")
        for p in extra_adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p):
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        adapter_config.device = device
    else:
        adapter_config = None
    print(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path)
    # Load tokenizer (unchanged)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    print(f"Parent Tokenizer: {len(tokenizer)}")
    
    ### GEMMA: Add pad token for CausalLM
    # CRITICAL FIX: Do NOT use eos_token as pad_token.
    # eos_token has learned embeddings that confuse the model when used as padding.
    # Gemma tokenizers have '<pad>' at index 0; we use that instead.
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        # Check if the tokenizer already knows about a '<pad>' token
        if '<pad>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<pad>'
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))
        config.pad_token_id = tokenizer.pad_token_id
        logger.warning(f"Set pad_token to '{tokenizer.pad_token}' (id={tokenizer.pad_token_id}), distinct from eos_token.")

    # --- LOG: Tokenizer summary ---
    logger.info("=" * 60)
    logger.info("[TOKENIZER] Vocab size: %d", len(tokenizer))
    logger.info("[TOKENIZER] pad_token: '%s' (id=%s)", tokenizer.pad_token, tokenizer.pad_token_id)
    logger.info("[TOKENIZER] eos_token: '%s' (id=%s)", tokenizer.eos_token, tokenizer.eos_token_id)
    logger.info("[TOKENIZER] bos_token: '%s' (id=%s)", tokenizer.bos_token, tokenizer.bos_token_id)
    logger.info("[TOKENIZER] pad ≠ eos: %s", tokenizer.pad_token_id != tokenizer.eos_token_id)
    logger.info("=" * 60)

    ### GEMMA: Load pretrained weights, then attach adapter modules
    # -----------------------------------------------------------------------
    # IMPORTANT: We ALWAYS load pretrained weights via from_pretrained.
    # `not_load_t5_checkpoint` originally meant "skip checkpoint loading for T5"
    # but for Gemma we always want the HuggingFace pretrained base weights.
    #
    # adapter_config is NOT a standard HF __init__ kwarg so we cannot pass it
    # to from_pretrained directly.  Instead we:
    #   1. First load the model with pretrained weights (adapter_config=None)
    #   2. Then call _init_adapter_modules() to wire up adapter sub-modules
    # -----------------------------------------------------------------------

    # Determine the checkpoint path
    last_checkpoint_path = training_args.output_dir
    resume_from_local = (
        not training_args.optimize_from_scratch
        and not training_args.optimize_from_scratch_with_loading_model
        and os.path.exists(os.path.join(last_checkpoint_path, "pytorch_model.bin"))
    )
    model_path = last_checkpoint_path if resume_from_local else model_args.model_name_or_path
    logger.warning("Loading student model from: %s", model_path)

    # Step 1: load pretrained weights into a model with NO adapters
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=True,  # safe: adapter keys simply won't be in the ckpt
    )

    # Step 2: attach adapter sub-modules (does NOT overwrite pretrained weights)
    model._init_adapter_modules(adapter_config)

    ### KD: Load the Teacher Model
    logger.info(f"Loading teacher model from {kd_args.teacher_model_name_or_path}")
    # Use standard GemmaForCausalLM for the teacher
    logger.info(model)
    logger.info(kd_args.teacher_model_name_or_path)
    teacher_model = StandardGemmaForConditionalGeneration.from_pretrained(
        kd_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        # Use a memory-efficient dtype for the teacher
        torch_dtype=torch.bfloat16, 
    )
    
    
    # Ensure teacher's pad token matches the tokenizer's (now distinct from eos)
    teacher_model.config.pad_token_id = tokenizer.pad_token_id

    # --- LOG: Model summary ---
    student_total = sum(p.numel() for p in model.parameters())
    student_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    teacher_total = sum(p.numel() for p in teacher_model.parameters())
    logger.info("=" * 60)
    logger.info("[MODEL] Student params: %s total, %s trainable (%.2f%%)",
                f"{student_total:,}", f"{student_train:,}",
                100.0 * student_train / student_total if student_total > 0 else 0)
    logger.info("[MODEL] Teacher params: %s total (frozen)", f"{teacher_total:,}")
    logger.info("[MODEL] Teacher pad_token_id: %s  |  Student pad_token_id: %s",
                teacher_model.config.pad_token_id, model.config.pad_token_id)
    logger.info("=" * 60)

    # set num_beams for evaluation (unchanged)
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # freezing the parameters.
    if training_args.do_train:
        ### WARNING: This function MUST be updated for Gemma's architecture
        logger.warning("Calling `freezing_params`. Ensure this function is updated for Gemma's architecture.")
        freezing_params(model, training_args, model_args, adapter_args)

    # Parameter logging (unchanged)
    if training_args.print_num_parameters:
        logger.info(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("Parameter name %s", name)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total parameters %s", total_params)
        
    # Gets the training/test/validation datasets.
    dataset_class = AutoTask
    if training_args.do_train:
        ### NOTE: The `add_prefix` arg might be T5-specific.
        # Your AutoTask class must handle this for Gemma.
        train_datasets = [dataset_class.get(task, seed=data_args.dataseed).get_dataset(
            split="train", n_obs=data_args.n_train, add_prefix=False if training_args.train_adapters else True)
            for task in data_args.tasks]
        dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
        train_dataset = datasets.concatenate_datasets(train_datasets)

        # --- LOG: Dataset sizes ---
        logger.info("=" * 60)
        for task_name, size in zip(data_args.tasks, dataset_sizes):
            logger.info("[DATA] Train task '%s': %d examples", task_name, size)
        logger.info("[DATA] Total training examples: %d", len(train_dataset))
        logger.info("=" * 60)

    training_args.remove_unused_columns = False
    
    eval_datasets = ({task: dataset_class.get(task, seed=data_args.dataseed).get_dataset(
        split="validation", n_obs=data_args.n_val,
        add_prefix=False if training_args.train_adapters else True,
        split_validation_test=training_args.split_validation_test)
                      for task in data_args.eval_tasks}
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None)
    
    test_dataset = (
        {task: dataset_class.get(task, seed=data_args.dataseed).get_dataset(
            split="test", n_obs=data_args.n_test,
            add_prefix=False if training_args.train_adapters else True,
            split_validation_test=training_args.split_validation_test)
            for task in data_args.eval_tasks} if training_args.do_test else None
    )
    
    # Defines the metrics for evaluation (unchanged)
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.eval_tasks, tokenizer) if training_args.predict_with_generate else None
    )
    
    ### WARNING: This TaskCollator MUST be updated for CausalLMs.
    # It must correctly create the 'labels' tensor from 'input_ids'
    # (e.g., by copying and masking prompt tokens with -100).
    data_collator = TaskCollator_gemma(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores)

    # --- LOG: Collator config ---
    logger.info("[COLLATOR] pad_token_id=%s, eos_token_id=%s, ignore_index=%s",
                data_collator.pad_token_id, data_collator.eos_token_id, data_collator.ignore_index)
    logger.info("[COLLATOR] max_source_length=%s, max_target_length=%s",
                data_collator.max_source_length, data_collator.max_target_length)

    ### GEMMA/KD: Defines the new trainer.
    trainer = GemmaAdapterKDTrainer(
        model=model,
        ### KD: Pass new arguments
        teacher_model=teacher_model,
        alpha_kd=kd_args.alpha_kd,
        temperature=kd_args.temperature_kd,
        
        # --- Original arguments ---
        config=config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        compute_metrics=None, # Using multi-task one
        multi_task_compute_metrics=compute_metrics_fn,
        data_args=data_args,
        tokenizer = tokenizer,
        dataset_sizes=dataset_sizes if training_args.do_train else None,
        adapter_config=adapter_config
    )

    # --- LOG: KD config ---
    logger.info("=" * 60)
    logger.info("[KD CONFIG] alpha_kd=%.4f, temperature=%.2f", kd_args.alpha_kd, kd_args.temperature_kd)
    logger.info("[KD CONFIG] learning_rate=%.6f", training_args.learning_rate)
    logger.info("=" * 60)

    if trainer.is_world_process_zero():
        arguments = get_training_args([model_args, data_args, training_args, adapter_args, kd_args]) # ### KD: Added kd_args
        handle_metrics("arguments", arguments, training_args.output_dir)

    # Trains the model (unchanged logic)
    if training_args.do_train:
        if trainer.is_world_process_zero():
            last_checkpoint_path = training_args.output_dir
            model_path = model_args.model_name_or_path if (training_args.optimize_from_scratch or not os.path.exists(os.path.join(last_checkpoint_path, 'pytorch_model.bin')))\
                else last_checkpoint_path
        
        if training_args.compute_time:
            torch.cuda.synchronize() 
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
        print(f"Teacher : {teacher_model}")
        print(f"Student : {model}")
        trainer.train(
            model_path=model_path \
                if (os.path.exists(training_args.output_dir) and not training_args.optimize_from_scratch) else None,
        )
        if training_args.compute_time: 
            torch.cuda.synchronize()
            end.record()
            total_time = {"total_time": start.elapsed_time(end)}
            print("###### total_time ", total_time)
        
        trainer.save_model()
        
        if trainer.is_world_process_zero():
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
            tokenizer.save_pretrained(training_args.output_dir)
            
    # Evaluation (Updated for Gemma and KD)
    all_metrics = {}
    if training_args.do_eval or training_args.do_test:
        if trainer.is_world_process_zero():
            
            last_checkpoint_path = get_last_checkpoint_path(training_args.output_dir)
            logger.info("Loading the latest checkpoint for evaluation.")
            ### GEMMA: Load Gemma config
            config = Gemma3TextConfig.from_pretrained(
                last_checkpoint_path,
                cache_dir=model_args.cache_dir)
            
            ### GEMMA: Load adapter-Gemma model
            model =   Gemma3ForConditionalGeneration.from_pretrained(
                last_checkpoint_path,
                from_tf=".ckpt" in training_args.output_dir,
                config=config,
                cache_dir=model_args.cache_dir,
                adapter_config=adapter_config
            )
            
            ### GEMMA/KD: Re-define the trainer with new classes and KD args
            trainer = GemmaAdapterKDTrainer(
                model=model,
                ### KD: Pass new arguments
                teacher_model=teacher_model, # Teacher model is already loaded
                alpha_kd=kd_args.alpha_kd,
                temperature=kd_args.temperature_kd,
                
                # --- Original arguments ---
                config=config,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_datasets,
                data_collator=data_collator, # Using the same (warned) collator
                compute_metrics=None,
                multi_task_compute_metrics=compute_metrics_fn,
                data_args=data_args,
                tokenizer = tokenizer,
                dataset_sizes=dataset_sizes if training_args.do_train else None,
                adapter_config=adapter_config
            )

        # This adapter logic is model-agnostic and should be fine
        if training_args.train_adapters:
            if adapter_args.adapter_config_name == "adapter" and data_args.adapters is not None:
                for name, sub_module in model.named_modules():
                    task_to_adapter = {eval_task: adapter for eval_task, adapter in
                                         zip(data_args.eval_tasks, data_args.adapters)}
                    if isinstance(sub_module, AdapterController):
                        sub_module.set_task_to_adapter_map(task_to_adapter)
    print("EVAL STARTS")
    # Evaluation loops (unchanged)
    if training_args.do_eval:
        metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            print("Yo")
            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)
    print("EVAL MID")
    if training_args.do_test:
        metrics = trainer.evaluate(test_dataset)
        if trainer.is_world_process_zero():
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)
    print("EVAL ENDS")
    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = torch.cuda.max_memory_allocated()/1024**2
        print(
            "Memory utilization",
            peak_memory,
            "MB"
        )
        memory_usage = {"peak_memory": peak_memory}
        print(f"memory usage : {memory_usage}")
    
    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()