import logging
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import utils
from hydra.utils import instantiate
from torch import nn
from typing import Union
from torch.utils import data as torch_data
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
import numpy as np
from itertools import islice
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils.data import Dataset
from llm import LLM
from evaluate import get_accuracy_webqsp
import pandas as pd
import json
import path_generate

separator = ">" * 30
line = "-" * 30

logger = logging.getLogger(__name__)

@torch.no_grad()
def ent_retrieval(
    cfg: DictConfig,
    model: nn.Module,
    data: Dataset,
    device: torch.device,
) -> list[dict]:
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    
    train_data, _, valid_data = data.qa_data
    graph = data.kg_data
    train_data = train_data or valid_data
        
    # Retrieve the supporting documents for each query
    sampler = torch_data.DistributedSampler(train_data, world_size, rank, shuffle=False)
    train_loader = torch_data.DataLoader(
        train_data, cfg.graph_retriever.retrieval_batch_size, sampler=sampler
    )

    

    model.eval()
    all_predictions: list[dict] = []
    for batch in tqdm(train_loader):
        batch = utils.cuda(batch, device = device)
        ent_pred = model(graph, batch, entities_weight = None)
        idx = batch["sample_id"]
        all_predictions.extend(
            {"id": i, "ent_pred": e}
            for i, e in zip(idx.cpu(), ent_pred.cpu())
        )
        
    # Gather the predictions across all processes 
    if utils.get_world_size() > 1:
        gathered_predictions = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_predictions, all_predictions)
    else:
        gathered_predictions = [all_predictions]  # type: ignore
        
    


    sorted_predictions = sorted(
        [item for sublist in gathered_predictions for item in sublist],  # type: ignore
        key=lambda x: x["id"],
    )

    
    utils.synchronize()
    print("sorted_predictions:", len(sorted_predictions))
    return sorted_predictions

def lora_llm(
    cfg: DictConfig,
    output_dir: str,
    model: nn.Module,
    train_datasets: dict,
    valid_datasets: dict,
    device: torch.device,
    batch_per_epoch: Union[int, None] = None,
) -> None:
    
    if cfg.train.num_epoch == 0:
        return
    
    torch.cuda.empty_cache()
    
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    
    train_dataloader_dict = {}
    for data_name, train_data in train_datasets.items():
        sampler = torch_data.DistributedSampler(train_data, world_size, rank)
        train_loader = torch_data.DataLoader(
            train_data, cfg.train.batch_size, sampler=sampler, drop_last=True, pin_memory=True, shuffle=False, collate_fn=utils.collate_fn
        )
        train_dataloader_dict[data_name] = train_loader
        batch_per_epoch = batch_per_epoch or len(train_loader)
        
    data_name_list = list(train_dataloader_dict.keys())
    
    # Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
            [{'params': params, 'lr': cfg.optimizer.lr, 'weight_decay': cfg.optimizer.wd}, ],
            betas=(0.9, 0.95)
        )
    trainable_params, all_param = model.print_trainable_params()
    if utils.get_rank() == 0:
        logger.warning(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model
        
    # Training
    best_result = float('inf')
    best_epoch = -1
    for i in range(cfg.train.num_epoch):
        epoch = i + 1
        parallel_model.train()
        
        if utils.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)
            
        epoch_loss, accum_loss = 0., 0.
        
        for dataloader in train_dataloader_dict.values():
            dataloader.sampler.set_epoch(epoch) 
        shuffled_data_name_list = np.random.permutation(
            data_name_list
        )  # Shuffle the dataloaders
        for data_name in shuffled_data_name_list:
            train_loader = train_dataloader_dict[data_name]
            step = 0
            for batch in tqdm(
                islice(train_loader, batch_per_epoch),
                desc=f"Training Batches: {epoch}",
                total=batch_per_epoch,
                disable=not utils.is_main_process(),
            ):
                batch = utils.cuda(batch, device=device)
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                
    
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
        
                if (step + 1) % cfg.train.log_interval == 0:
                    utils.adjust_learning_rate(optimizer.param_groups[0], cfg.optimizer.lr, step / len(train_loader) + epoch, cfg)
                
                optimizer.step()
                epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
    
                if (step + 1) % cfg.train.log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    if utils.get_rank() == 0:
                        logger.warning(f"Lr: {lr:g}")
                        logger.warning(f"Accum Loss: {accum_loss / cfg.train.log_interval :g}")
                    accum_loss = 0.
                
                step += 1
                torch.cuda.empty_cache()

        # Clear cache after each epoch to prevent memory buildup
        torch.cuda.empty_cache()
        
        if utils.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning(
                f"Epoch: {epoch}|{cfg.train.num_epoch}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader):g}"
            )
            
        utils.synchronize()
        torch.cuda.empty_cache()
        
        if cfg.train.do_eval:
            if utils.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Evaluate on valid")
            result = validate(cfg, model, valid_datasets, epoch, device=device)
        else:
            result = float("inf")
            best_result = float("-inf")
        if utils.get_rank() == 0:
            if result < best_result:
                best_result = result
                best_epoch = epoch
                
                logger.warning("Save checkpoint")
                utils.save_checkpoint(model, optimizer, epoch, output_dir, cfg, is_best=True)
                
                
                # state = {
                #     "model": model.state_dict(),
                #     "optimizer": optimizer.state_dict(),
                # }
                # logger.warning("Save checkpoint to llm_model_best.pth")
                # torch.save(state, os.path.join(output_dir, "llm_model_best.pth"))
                # logger.warning("Save completed!")
                
            logger.warning(f"Best loss: {best_result:g} at epoch {best_epoch}")

            if epoch - best_epoch >= cfg.train.patience:
                print(f'Early stop at epoch {epoch}')
                break
    
    # if utils.get_rank() == 0:
    #     logger.warning("Load checkpoint from llm_model_best.pth")
    # if dist.is_initialized():
    #     dist.barrier() 
    # state = torch.load(
    #     os.path.join(output_dir, "llm_model_best.pth"),
    #     map_location=device,
    #     weights_only=False,
    # )
    # model.load_state_dict(state["model"])
    # utils.synchronize()        

    # Clear GPU cache after loading the best model
    torch.cuda.empty_cache()
        
        
@torch.no_grad()
def validate(
    cfg: DictConfig,
    model: nn.Module,
    valid_datasets: dict,
    epoch: int,
    device: torch.device,
) -> float:
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    
    for data_name, valid_data in valid_datasets.items():
        sampler = torch_data.DistributedSampler(valid_data, world_size, rank)
        valid_loader = torch_data.DataLoader(
            valid_data, cfg.train.batch_size, sampler=sampler, drop_last=False, pin_memory=True, shuffle=False, collate_fn=utils.collate_fn
        )
        valid_loss = 0.
        eval_output = []
        model.eval()
            
        for batch in tqdm(
            valid_loader,
            desc = f"Validating {data_name}",
            disable = not utils.is_main_process(),
        ):
            batch = utils.cuda(batch, device = device)
            loss = model(batch)
            valid_loss += loss.item()    
        valid_loss = valid_loss/len(valid_loader) 
        
        if utils.get_rank() == 0:
            logger.warning(f"Epoch: {epoch}|{cfg.train.num_epoch}: Valid Loss: {valid_loss}")
    
    utils.synchronize()
    torch.cuda.empty_cache()  
    return valid_loss         
    
# @torch.no_grad()
# def evaluate(
#     cfg: DictConfig,
#     model: nn.Module,
#     output_dir: str,
#     data_name: str,
#     samples: list,
#     device: torch.device,
# ) -> None:
#     world_size = utils.get_world_size()
#     rank = utils.get_rank()
    
#     sampler = torch_data.DistributedSampler(samples, world_size, rank, shuffle=False)
#     inference_loader = torch_data.DataLoader(
#         samples, cfg.valid.inference_batch_size, sampler=sampler, drop_last=False, pin_memory=True, shuffle=False, collate_fn=utils.collate_fn
#     )
    
#     model.eval()
#     progress_bar_test = tqdm(range(len(inference_loader)))
#     with open(os.path.join(output_dir, f"{data_name}_prediction.csv"), "w") as f:
#         for _, batch in enumerate(inference_loader):
#             batch = utils.cuda(batch, device = device)
#             with torch.no_grad():
#                 output = model.inference(batch)
#                 df = pd.DataFrame(output)
#                 for _, row in df.iterrows():
#                     f.write(json.dumps(dict(row)) + "\n")
#             progress_bar_test.update(1)

#     torch.cuda.empty_cache()  # Clear cache after evaluation
#     return os.path.join(output_dir, f"{data_name}_prediction.csv") 
    

@hydra.main(config_path = "config", config_name = "train_llm", version_base = None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode(cfg.train.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        
    # Build models    
    device = utils.get_device()
    
    model, model_config = utils.load_model_from_pretrained(
        cfg.graph_retriever.model_path
    )
    model = model.to(device)
    
    llm_model = LLM(cfg.llm.llm_model_path, cfg.llm.llm_frozen, device)
    llm_model.to(device) 
    
    # Data preparation
    train_datasets = {}
    valid_datasets = {}    
    datasets = utils.get_multi_dataset(cfg)
    for data_name, data in datasets.items():
        if data_name not in cfg.datasets.train_names + cfg.datasets.valid_names:
            raise ValueError(f"Unknown data name: {data_name}")
        data.kg_data = data.kg_data.to(device)
        # Ent retrieval
        retrieval_result = ent_retrieval(cfg, model, data, device=device)   
        rel_emb = data.kg_data.rel_emb
        ent_emb = data.ent_emb
        graph_list = data.raw_graph
        ent2id = data.ent2id
        rel2id = data.rel2id
        train_data, _, valid_data = data.qa_data
            
        
        if data_name in cfg.datasets.train_names:
            raw_train_data = data.raw_train_data
            samples = []
            inference_path = []
            print("Generating inference path……")
            for i, item in enumerate(retrieval_result):
                path = path_generate.retrieval_path(graph_list[i], item["ent_pred"], rel_emb, ent_emb, ent2id, rel2id, raw_train_data[i]["q_entity"], train_data[i]["question_embeddings"])     
                inference_path.append(path)
            for i in range(len(inference_path)):
                samples.append({
                    "id": raw_train_data[i]["id"],
                    "question": raw_train_data[i]["question"],
                    "answer": ('|').join(raw_train_data[i]["answer"]).lower(),
                    "inference_path": inference_path[i]
                })
            
            train_datasets[data_name]= samples
            
        if data_name in cfg.datasets.valid_names:
            raw_valid_data = data.raw_valid_data
            samples = []
            inference_path = []
            print("Generating inference path……")
            for i, item in enumerate(retrieval_result):
                path = path_generate.retrieval_path(graph_list[i], item["ent_pred"], rel_emb, ent_emb, ent2id, rel2id, raw_valid_data[i]["q_entity"], valid_data[i]["question_embeddings"]) 
                inference_path.append(path)
            for i in range(len(inference_path)):
                samples.append({
                    "id": raw_valid_data[i]["id"],
                    "question": raw_valid_data[i]["question"],
                    "answer": ('|').join(raw_valid_data[i]["answer"]).lower(),
                    "inference_path": inference_path[i]
                })
            
            valid_datasets[data_name]= samples

    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(cfg.train.checkpoint, map_location="cpu")
            llm_model.load_state_dict(state["model"])
        # Try to load the model from the remote dictionary
        else:
            llm_model, _ = utils.load_model_from_pretrained(cfg.train.checkpoint)
    
    lora_llm(
        cfg,
        output_dir,
        llm_model,
        train_datasets,
        valid_datasets,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )
    
     # Save the model into the format for QA inference
    
    # # Evaluation
    # if utils.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning("Evaluate on valid")
    #     for data_name, samples in valid_datasets.items():
    #         logger.warning(f"Evaluate on {data_name}")
    #         path = evaluate(cfg, llm_model, output_dir, data_name, samples, device=device)  
    #         acc, hit, f1, pre, recall, num = get_accuracy_webqsp(path)
    #         logger.warning(f"Accuracy: {acc / 100:.4f}, Hit: {hit:.4f}, Precision: {pre / 100:.4f}, Recall: {recall / 100:.4f},F1: {f1 / 100:.4f}")   
   
if __name__ == "__main__":
    main()
