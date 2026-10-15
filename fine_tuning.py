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
from tqdm import tqdm
import os
import numpy as np
from itertools import islice
from torch.nn import functional as F
from torch import distributed as dist
import wandb

logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30

def train_and_validate(
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
    
     # Create dataloader for each dataset
    train_dataloader_dict = {}
    for data_name, dataset in train_datasets.items():
        train_data = dataset["data"]
        sampler = torch_data.DistributedSampler(train_data, world_size, rank)
        train_loader = torch_data.DataLoader(
            train_data, cfg.train.batch_size, sampler=sampler
        )
        train_dataloader_dict[data_name] = train_loader
        batch_per_epoch = batch_per_epoch or len(train_loader)
        
    data_name_list = list(train_dataloader_dict.keys())

    optimizer = instantiate(cfg.optimizer, model.parameters())

    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    # Initialize Losses
    loss_fn_list = []
    for loss_cfg in cfg.task.losses:
        loss_fn = instantiate(loss_cfg.loss)
        loss_fn_list.append(
            {
                "name": loss_cfg.name,
                "loss_fn": loss_fn,
                **loss_cfg.cfg,
            }
        )
        
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    best_result = float("-inf")
    best_epoch = -1
    
    batch_id = 0
    for i in range(0, cfg.train.num_epoch):
        epoch = i + 1
        parallel_model.train()

        if utils.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)

        losses: dict[str, list] = {loss_dict["name"]: [] for loss_dict in loss_fn_list}
        losses["loss"] = []
        
        for dataloader in train_dataloader_dict.values():
            dataloader.sampler.set_epoch(epoch)
        # np.random.seed(epoch)  # TODO: should we use the same dataloader for all processes?
        shuffled_data_name_list = np.random.permutation(
            data_name_list
        )  # Shuffle the dataloaders
        for data_name in shuffled_data_name_list:
            train_loader = train_dataloader_dict[data_name]
            graph = train_datasets[data_name]["graph"]
            entities_weight = None
            # TODO: how to calculate the weight?
            # if cfg.train.init_entities_weight:
            #     entities_weight = utils.get_entities_weight(ent2docs)
            for batch in tqdm(
                islice(train_loader, batch_per_epoch),
                desc=f"Training Batches: {epoch}",
                total=batch_per_epoch,
                disable=not utils.is_main_process(),
            ):
                batch = utils.cuda(batch, device=device)
                pred = parallel_model(graph, batch, entities_weight=entities_weight)
                target = batch["answer_entities_masks"] 
            
                loss = 0
                tmp_losses = {}
                for loss_dict in loss_fn_list:
                    loss_fn = loss_dict["loss_fn"]
                    weight = loss_dict["weight"]
                    single_loss = loss_fn(pred, target)
                    tmp_losses[loss_dict["name"]] = single_loss.item()
                    loss += weight * single_loss
                tmp_losses["loss"] = loss.item()  
                    
                loss.backward()  
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                for loss_log in tmp_losses:
                    losses[loss_log].append(tmp_losses[loss_log])
                        

                if utils.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    for loss_log in tmp_losses:
                        logger.warning(f"{loss_log}: {tmp_losses[loss_log]:g}")
                        # wandb.log({f"train/{loss_log}": tmp_losses[loss_log]})
                batch_id += 1    
                
        if utils.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            for loss_log in losses:
                avg_loss = sum(losses[loss_log]) / len(losses[loss_log])
                wandb.log({f"train/{loss_log}": avg_loss}, step=epoch)
                logger.warning(f"Avg: {loss_log}: {avg_loss:g}")
    
        utils.synchronize()
        torch.cuda.empty_cache()

        if cfg.train.do_eval:
            if rank == 0:
                logger.warning(separator)
                logger.warning("Evaluate on valid")
            result = validate(cfg, model, valid_datasets, device=device)
            wandb.log({"valid/mrr": result}, step=epoch)
        else:
            result = float("inf")
            best_result = float("-inf")
        if rank == 0:
            if result > best_result:
                best_result = result
                best_epoch = epoch
                logger.warning("Save checkpoint to model_best.pth")
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(output_dir, "model_best.pth"))
            logger.warning(f"Best mrr: {best_result:g} at epoch {best_epoch}")
            
            if epoch - best_epoch >= cfg.train.patience:
                print(f'Early stop at epoch {epoch}')
                break

    if rank == 0:
        logger.warning("Load checkpoint from model_best.pth")
    if dist.is_initialized():
        dist.barrier() 
    state = torch.load(
        os.path.join(output_dir, "model_best.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    utils.synchronize()
    
    wandb.finish()
                


@torch.no_grad()
def validate(
    cfg: DictConfig,
    model: nn.Module,
    valid_datasets: dict,
    device: torch.device,
    return_metrics: bool = False,
) -> Union[float, dict]:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    all_metrics = {}
    all_mrr = []
    for data_name, data in valid_datasets.items():
        valid_data = data["data"]
        graph = data["graph"]
        sampler = torch_data.DistributedSampler(valid_data, world_size, rank)
        valid_loader = torch_data.DataLoader(
            valid_data, cfg.train.batch_size, sampler=sampler
        )

        model.eval()
        ent_preds = []
        ent_targets = []
        
        entities_weight = None
        
        for batch in tqdm(
            valid_loader,
            desc = f"Validating {data_name}",
            disable = not utils.is_main_process(),
        ):
            batch = utils.cuda(batch, device = device)
            ent_pred = model(graph, batch, entities_weight = entities_weight)
            target_entities_mask = batch[
                "answer_entities_masks"
            ] 
            target_entities = target_entities_mask.bool()
            ent_ranking, target_ent_ranking = utils.batch_evaluate(
                ent_pred, target_entities
            )
            
            # answer set cardinality prediction
            ent_prob = F.sigmoid(ent_pred)
            num_pred = (ent_prob * (ent_prob > 0.5)).sum(dim=-1)
            num_target = target_entities_mask.sum(dim=-1)
            ent_preds.append((ent_ranking, num_pred))
            ent_targets.append((target_ent_ranking, num_target))
            
        ent_pred = utils.cat(ent_preds)
        ent_target = utils.cat(ent_targets)
        
        ent_pred, ent_target = utils.gather_results(
            ent_pred, ent_target, rank, world_size, device
        )
        
        ent_metrics = utils.evaluate(ent_pred, ent_target, cfg.task.metric)
        metrics = {}
        if rank == 0:
            for key, value in ent_metrics.items():
                metrics[f"ent_{key}"] = value
            metrics["mrr"] = ent_metrics["mrr"]
            logger.warning(f"{'-' * 15} Validate on {data_name} {'-' * 15}")
            utils.print_metrics(metrics, logger)
        else:
            metrics["mrr"] = ent_metrics["mrr"]
        all_metrics[data_name] = metrics
        all_mrr.append(metrics["mrr"])
    utils.synchronize()
    all_avg_mrr = np.mean(all_mrr)
    return all_avg_mrr if not return_metrics else metrics
            
   
@hydra.main(config_path = "config", config_name = "fine_tuning", version_base = None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode(cfg.train.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        
    wandb.init(
        project="Rag",
        name=f"Run_{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
        
    datasets = utils.get_multi_dataset(cfg)
    device = utils.get_device()
    
    rel_emb_dim = {data.rel_emb_dim for data in datasets.values()}
    assert (
        len(rel_emb_dim) == 1
    ), "All datasets should have the same relation embedding dimension"
    
    model = instantiate(cfg.model, rel_emb_dim=rel_emb_dim.pop())
    
    train_datasets = {}
    valid_datasets = {}
    
    for data_name, data in datasets.items():
        if data_name not in cfg.datasets.train_names + cfg.datasets.valid_names:
            raise ValueError(f"Unknown data name: {data_name}")
    
        train_data, _, valid_data = data.qa_data
        graph = data.kg_data.to(device)
    
        if data_name in cfg.datasets.train_names:
            train_datasets[data_name] = {
                "data": train_data,
                "graph": graph,
            }
        if data_name in cfg.datasets.valid_names:
            valid_datasets[data_name] = {
                "data": valid_data,
                "graph": graph,
            }
    
    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(cfg.train.checkpoint, map_location="cpu")
            model.load_state_dict(state["model"])
        # Try to load the model from the remote dictionary
        else:
            model, _ = utils.load_model_from_pretrained(cfg.train.checkpoint)

    model = model.to(device)
    train_and_validate(
        cfg,
        output_dir,
        model,
        train_datasets,
        valid_datasets,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )
    
    if cfg.train.do_eval:
        if utils.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        validate(cfg, model, valid_datasets, device=device)
    
    # Save the model into the format for QA inference
    if (
        utils.is_main_process()
        and cfg.train.save_pretrained
        and cfg.train.num_epoch > 0
    ):
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        utils.save_model_to_pretrained(model, cfg, pre_trained_dir)

    utils.synchronize()
    utils.cleanup()
    
if __name__ == "__main__":
    main()
