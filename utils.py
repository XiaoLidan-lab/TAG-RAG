from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_class
import os
import torch
from torch import distributed as dist
from ultra import variadic
from torch_scatter import scatter_add
import json
import datetime
from typing import Any
from hydra.utils import instantiate
from torch_geometric.data import Batch
import math

def get_multi_dataset(cfg: DictConfig) -> dict:
    data_name_list = []
    # Remove duplicates in the list
    for data_name in cfg.datasets.train_names + cfg.datasets.valid_names:
        if data_name not in data_name_list:
            data_name_list.append(data_name)
    dataset_cls = get_class(cfg.datasets._target_)
    dataset_list = {}
    for data_name in data_name_list:
        data = dataset_cls(**cfg.datasets.cfgs, data_name=data_name)
        dataset_list[data_name] = data
    return dataset_list

# get current process rank
def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(get_local_rank())
    else:
        device = torch.device("cpu")
    return device

def is_main_process() -> bool:
    return get_rank() == 0

def save_model_to_pretrained(
    model: torch.nn.Module, cfg: DictConfig, path: str
) -> None:
    os.makedirs(path, exist_ok=True)
    model_config = OmegaConf.to_container(cfg.model)
    model_config["rel_emb_dim"] = model.rel_emb_dim
    config = {
        "model_config": model_config,
    }

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    torch.save({"model": model.state_dict()}, os.path.join(path, "model.pth"))
    
def synchronize() -> None:
    if get_world_size() > 1:
        dist.barrier()
        
def cleanup() -> None:
    if get_world_size() > 1:
        dist.destroy_process_group()
        
def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1

def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0

def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)

    raise TypeError(f"Can't transfer object type `{type(obj)}`")

def batch_evaluate(pred, target, limit_nodes=None):
    num_target = target.sum(dim=-1)
    # answer2query = functional._size_to_index(num_answer)
    answer2query = torch.repeat_interleave(num_target)

    num_entity = pred.shape[-1]

    # in inductive (e) fb_ datasets, the number of nodes in the graph structure might exceed
    # the actual number of nodes in the graph, so we'll mask unused nodes
    if limit_nodes is not None:
        # print(f"Keeping only {len(limit_nodes)} nodes out of {num_entity}")
        keep_mask = torch.zeros(num_entity, dtype=torch.bool, device=limit_nodes.device)
        keep_mask[limit_nodes] = 1
        # keep_mask = F.one_hot(limit_nodes, num_entity)
        pred[:, ~keep_mask] = float("-inf")

    order = pred.argsort(dim=-1, descending=True)

    range = torch.arange(num_entity, device=pred.device)
    ranking = scatter_add(range.expand_as(order), order, dim=-1)
    
    target_ranking = ranking[target]
    # unfiltered rankings of all answers
    order_among_answer = variadic.variadic_sort(target_ranking, num_target)[1]
    order_among_answer = (
        order_among_answer + (num_target.cumsum(0) - num_target)[answer2query]
    )
    ranking_among_answer = scatter_add(
        variadic.variadic_arange(num_target), order_among_answer
    )

    # filtered rankings of all answers
    ranking = target_ranking - ranking_among_answer + 1
    ends = num_target.cumsum(0)
    starts = ends - num_target
    hard_mask = variadic.multi_slice_mask(starts, ends, ends[-1])
    # filtered rankings of hard answers
    ranking = ranking[hard_mask]

    return ranking, target_ranking

def cat(objs, *args, **kwargs):
    """
    Concatenate a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.cat(objs, *args, **kwargs)
    elif isinstance(obj, dict):
        return {k: cat([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cat(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError(f"Can't perform concatenation over object type `{type(obj)}`")

def gather_results(pred, target, rank, world_size, device):
    # for multi-gpu setups: join results together
    # for single-gpu setups: doesn't do anything special
    ranking, num_pred = pred
    answer_ranking, num_target = target

    all_size_r = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_ar = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_p = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_r[rank] = len(ranking)
    all_size_ar[rank] = len(answer_ranking)
    all_size_p[rank] = len(num_pred)
    if world_size > 1:
        dist.all_reduce(all_size_r, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_ar, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_p, op=dist.ReduceOp.SUM)

    # obtaining all ranks
    cum_size_r = all_size_r.cumsum(0)
    cum_size_ar = all_size_ar.cumsum(0)
    cum_size_p = all_size_p.cumsum(0)

    all_ranking = torch.zeros(all_size_r.sum(), dtype=torch.long, device=device)
    all_num_pred = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)
    all_answer_ranking = torch.zeros(all_size_ar.sum(), dtype=torch.long, device=device)
    all_num_target = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)

    all_ranking[cum_size_r[rank] - all_size_r[rank] : cum_size_r[rank]] = ranking
    all_num_pred[cum_size_p[rank] - all_size_p[rank] : cum_size_p[rank]] = num_pred
    all_answer_ranking[cum_size_ar[rank] - all_size_ar[rank] : cum_size_ar[rank]] = (
        answer_ranking
    )
    all_num_target[cum_size_p[rank] - all_size_p[rank] : cum_size_p[rank]] = num_target

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_answer_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_target, op=dist.ReduceOp.SUM)

    return (all_ranking.cpu(), all_num_pred.cpu()), (
        all_answer_ranking.cpu(),
        all_num_target.cpu(),
    )
    
def evaluate(pred, target, metrics):
    ranking, num_pred = pred
    answer_ranking, num_hard = target

    metric = {}
    for _metric in metrics:
        if _metric == "mrr":
            answer_score = 1 / ranking.float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric.startswith("recall@"):
            threshold = int(_metric[7:])
            answer_score = (ranking <= threshold).float()
            query_score = (
                variadic.variadic_sum(answer_score, num_hard) / num_hard.float()
            )
        elif _metric.startswith("hits@"):
            threshold = int(_metric[5:])
            answer_score = (ranking <= threshold).float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric == "mape":
            query_score = (num_pred - num_hard).abs() / (num_hard).float()
        else:
            raise ValueError(f"Unknown metric `{_metric}`")

        score = query_score.mean()
        name = _metric
        metric[name] = score.item()

    return metric

def print_metrics(metrics, logger, roundto=4):
    order = sorted(list(metrics.keys()))
    for key in order:
        logger.warning(f"{key}: {round(metrics[key], roundto)}")
        
        
def init_distributed_mode(timeout: None = None) -> None:
    world_size = get_world_size()
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(get_local_rank())
        if timeout is not None:
            timeout = datetime.timedelta(minutes=timeout)
        dist.init_process_group("nccl", init_method="env://", timeout=timeout)
        synchronize()
        setup_for_distributed(get_rank() == 0)
        
def setup_for_distributed(is_master: bool) -> None:
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def load_model_from_pretrained(path: str) -> tuple[torch.nn.Module, dict]:
    config_path = os.path.join(path, "config.json")
    if config_path is None:
        raise FileNotFoundError(f"config.json not found in {path}")
    with open(config_path) as f:
        config = json.load(f)
    model = instantiate(config["model_config"])
    model_path = os.path.join(path, "model.pth")
    if model_path is None:
        raise FileNotFoundError(f"model.pth not found in {path}")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model, config


def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    return batch

def adjust_learning_rate(param_group, LR, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-6
    if epoch < cfg.optimizer.warmup_epoch:
        lr = LR * epoch / cfg.optimizer.warmup_epoch
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - cfg.optimizer.warmup_epoch) / (cfg.train.num_epoch - cfg.optimizer.warmup_epoch)))
    param_group["lr"] = lr
    return lr

def save_checkpoint(model, optimizer, cur_epoch, output_dir, cfg, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "epoch": cur_epoch,
    }
    path = os.path.join(output_dir, "llm_model_best.pth")
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)

def reload_best_model(model, cfg):
    """
    Load the best checkpoint for evaluation.
    """

    checkpoint = torch.load(cfg.llm.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model