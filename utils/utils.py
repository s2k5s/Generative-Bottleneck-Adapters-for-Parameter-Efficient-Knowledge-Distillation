import glob
import os
from dataclasses import asdict
from logging import getLogger
from third_party.utils import (
    assert_all_frozen,
    freeze_embeds,
    freeze_params,
    save_json)
from third_party.models.modeling_gemma import Gemma3RMSNorm

from adapters import (AdapterController, MetaAdapterController, 
                              AdapterLayersHyperNetController, AdapterLayersOneHyperNetController)
from data import TASKS
import torch
import torch.fx as fx

logger = getLogger(__name__)


def create_dir(output_dir):
    """Create a directory if it does not already exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def handle_metrics(split, metrics, output_dir):
    """Log metrics and save them to disk for a given split."""
    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json_file(metrics, f"{split}_results.json", output_dir)


def save_json_file(json_dict, outfile_name, output_dir):
    """Save a dictionary to disk as JSON under the provided output directory."""
    save_json(json_dict, os.path.join(output_dir, outfile_name))


def get_training_args(arguments_list):
    """
    Concatenate all training arguments except evaluation strategy which
    is not Json serializable.
    Args:
        arguments_list: list of dataclasses.
    Return:
        arguments: concatenated arguments.
    """
    all_arguments = {}
    for arguments in arguments_list:
        all_arguments.update(asdict(arguments))
    all_arguments.pop("evaluation_strategy")
    return all_arguments


def get_last_checkpoint_path(output_dir):
    """
    Finds the path for the last checkpoint saved in the output_dir
    Args:
        output_dir:  output_dir
    Returns:
        path to the last checkpoint saved in the output dir.
    """
    paths = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if len(paths) == 0:
        return output_dir
    else:
        checkpoints = [int(checkpoint.split('-')[-1]) for checkpoint in paths]
        max_checkpoint = max(checkpoints)
        return os.path.join(output_dir, "checkpoint-" + str(max_checkpoint))


def use_task_specific_params(model, task):
    """Update config with task specific params during evaluation."""
    task_dataset = TASKS[task]
    task_specific_config = task_dataset.task_specific_config
    if task_specific_config is not None:
        logger.info(f"using task specific params for {task}: {task_specific_config}")
        model.config.update(task_specific_config)


def reset_config(model, config):
    """Resets the config file to the one provided."""
    model.config = config
    logger.info(f"config is reset to the initial values.")

def write_model_status_to_file(model, file_path):
    """Write a compact report of trainable and frozen parameters to disk."""
    with open(file_path, "w") as f:
        f.write(f"{'Parameter Name':<70} | {'Status'}\n")
        f.write("-" * 85 + "\n")
        
        for name, param in model.named_parameters():
            status = "FROZEN" if not param.requires_grad else "TRAINABLE"
            f.write(f"{name} | {tuple(param.shape)}| {status}\n")
        
        # Add a summary at the end
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        f.write("\n" + "="*30 + "\n")
        f.write(f"SUMMARY:\n")
        f.write(f"Trainable Params: {trainable:,}\n")
        f.write(f"Frozen Params:    {frozen:,}\n")
        f.write(f"Total Params:     {trainable + frozen:,}\n")

def write_trainable_only(model, file_path):
    """Write the names of trainable parameters to a text file."""
    with open(file_path, "w") as f:
        f.write("--- LAYERS CURRENTLY BEING TRAINED ---\n\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"{name} ({param.numel()} parameters)\n")

def graph_print(model, file_path):
    """Placeholder for optional graph export logic.

    This helper is intentionally left as a no-op for now so the training code
    can stay focused on the core adapter and KD workflow.
    """
    pass


def freezing_params(model, training_args, model_args, adapter_args):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      training_args: defines the training arguments.
      model_args: defines the model arguments.
      adapter_args: defines the adapters arguments.
    """
    
    # If we are training adapters, we freeze all parameters except the
    # parameters of computing task embeddings and adapter controllers.
    if training_args.train_adapters:
        freeze_params(model)
        logger.info("Freezing base model parameters and enabling adapter updates.")
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (MetaAdapterController, AdapterController)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
        if adapter_args.adapter_config_name == "meta-adapter":
            logger.info("Unfreezing task embedding controller for meta-adapter training.")
            for param in model.task_embedding_controller.parameters():
                param.requires_grad = True
        if adapter_args.unique_hyper_net:
            logger.info("Enabling unique hyper-network parameters for adapter training.")
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (AdapterLayersHyperNetController, AdapterController)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        if adapter_args.efficient_unique_hyper_net:
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (AdapterLayersOneHyperNetController)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

    if model_args.freeze_model:
        freeze_params(model)

    # Freezes all models parameters except last linear layer of decoder.
    if model_args.freeze_model_but_lm_head:
        freeze_params(model)
        for param in model.lm_head.parameters():
            param.requires_grad = True

    if model_args.freeze_embeds:
        freeze_embeds(model)

    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    # In case of meta-adapters and if task-embeddings are paramteric,
    # freezes all parameters except task-embedding parameters.
    if model_args.freeze_model_but_task_embeddings:
        freeze_params(model)
        if adapter_args.adapter_config_name == "meta-adapter" and \
                not isinstance(model.task_embedding_controller.task_to_embeddings, dict):
            for param in model.task_embedding_controller.task_to_embeddings.parameters():
                param.requires_grad = True

    # Unfreezes last linear layer of decoder.
    if model_args.unfreeze_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # Unfreezes layer norms.
    model_args.unfreeze_layer_norms = False
    if model_args.unfreeze_layer_norms:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, Gemma3RMSNorm):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

    if model_args.unfreeze_model:
        for param in model.parameters():
            param.requires_grad = True

    write_model_status_to_file(model, "model_1.txt")
    write_trainable_only(model, "model_2.txt")
    with open("model_3.txt", "w") as f: f.write(str(model))
    graph_print(model, "/model_4.txt" )

    