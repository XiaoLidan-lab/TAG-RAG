import logging
import json
import os 
from torch import nn
from torch.utils.data import Dataset
from torch.utils import data as torch_data
from tqdm import tqdm
import torch
from hydra.utils import instantiate
from typing import Union
import utils
import pandas as pd

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from llm import LLM

from evaluate import get_accuracy_webqsp
from hydra.utils import get_class
import path_generate
from torch import distributed as dist



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
    
    _, test_data, _ = data.qa_data
    graph = data.kg_data
    
    # Retrieve the supporting documents for each query
    sampler = torch_data.DistributedSampler(test_data, world_size, rank, shuffle=False)
    test_loader = torch_data.DataLoader(
        test_data, cfg.test.retrieval_batch_size, sampler=sampler
    )
    

    model.eval()
    all_predictions: list[dict] = []
    for batch in tqdm(test_loader):
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
    return sorted_predictions


def ans_prediction(
    cfg: DictConfig,
    output_dir: str,
    data: Dataset,
    device: torch.device,
    retrieval_result: list[dict],
    data_name: str
) -> str:
    
    # input retrieved entities
    
    # test_data = data.raw_test_data
    # id2ent = {v: k for k, v in qa_data.ent2id.items()}
    
    # retrieved_ent_list = []
    # for item in retrieval_result:
    #     retrieved_ent_idx = torch.topk(
    #         item["ent_pred"], cfg.test.save_top_k_entity, dim = -1
    #     ).indices
    #     retrieved_ent = [id2ent[i.item()].lower() for i in retrieved_ent_idx]
    #     retrieved_ent_list.append(retrieved_ent)

    # samples = []
    
    # for i in range(len(retrieved_ent_list)):
    #     samples.append({
    #         "id": test_data[i]["id"],
    #         "question": f'Question: {test_data[i]["question"]}\nAnswer: ',
    #         "answer": ('|').join(test_data[i]["answer"]).lower(),
    #         "ent": retrieved_ent_list[i]
    #     })
        
        
        
    # input retrieved inference path
    
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    
    # model = LLM(cfg.llm.llm_model_path, cfg.llm.llm_frozen, device)
    # model.to(device)
    
    model = LLM(cfg.llm.llm_model_path, False, device)
    model.to(device)
 
    logger.warning("Loading the checkpoint……")
    model = utils.reload_best_model(model, cfg)
    
    
    raw_test_data = data.raw_test_data
    rel_emb = data.kg_data.rel_emb
    ent_emb = data.ent_emb
    graph_list = data.raw_graph
    ent2id = data.ent2id
    rel2id = data.rel2id
    _, test_data, _ = data.qa_data
    
    samples = []
    inference_path = []
    print("Generating inference path……")
    for i, item in enumerate(retrieval_result):
        path = path_generate.retrieval_path(graph_list[i], item["ent_pred"], rel_emb, ent_emb, ent2id, rel2id, raw_test_data[i]["q_entity"], test_data[i]["question_embeddings"])   
        inference_path.append(path)
        
        
    for i in range(len(inference_path)):
        samples.append({
            "id": raw_test_data[i]["id"],
            "question": f'Question: {raw_test_data[i]["question"]}\nAnswer: ',
            "answer": ('|').join(raw_test_data[i]["answer"]).lower(),
            "inference_path": inference_path[i]
        }) 
        
    
    
    
    sampler = torch_data.DistributedSampler(samples, world_size, rank, shuffle=False)
    inference_loader = torch_data.DataLoader(
        samples, cfg.test.inference_batch_size, sampler=sampler, drop_last=False, pin_memory=True, shuffle=False, collate_fn=utils.collate_fn
    )
    

    model.eval()
    progress_bar_test = tqdm(range(len(inference_loader)))
    with open(os.path.join(output_dir, f"{data_name}_prediction.csv"), "w") as f:
        for _, batch in enumerate(inference_loader):
            batch = utils.cuda(batch, device = device)
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    return os.path.join(output_dir, f"{data_name}_prediction.csv")        
               

        

@hydra.main(config_path = "config", config_name = "inference", version_base = None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode(cfg.test.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        
    model, model_config = utils.load_model_from_pretrained(
        cfg.graph_retriever.model_path
    )
    
    acc_list =[]
    hit_list = []
    f1_list = []
    pre_list = []
    recall_list = []
    num_list = []
    for data_name in cfg.datasets.data_names:
        dataset_cls = get_class(cfg.datasets._target_)
        data = dataset_cls(**cfg.datasets.cfgs, data_name=data_name)
        device = utils.get_device()
        model = model.to(device)
        data.kg_data = data.kg_data.to(device)
        
        
        retrieval_result = ent_retrieval(cfg, model, data, device=device)
        
        if utils.is_main_process():
            if cfg.test.save_retrieval and retrieval_result is not None:
                logger.info(
                    f"Ranking saved to disk: {os.path.join(output_dir, data_name + '_retrieval_result.pt')}"
                )
                torch.save(
                    retrieval_result, os.path.join(output_dir, data_name + '_retrieval_result.pt')
                )
            ans_prediction(cfg, output_dir, data, device, retrieval_result, data_name)
        
        # Evaluation
        acc, hit, f1, pre, recall, num = get_accuracy_webqsp(os.path.join(output_dir, f"{data_name}_prediction.csv"))
        acc_list.append(acc)
        hit_list.append(hit)
        f1_list.append(f1)
        pre_list.append(pre)
        recall_list.append(recall)
        num_list.append(num)
        logger.warning(f"{data_name}:Accuracy: {acc / num:.4f}, Hit: {hit / num:.4f}, Precision: {pre/ num:.4f}, Recall: {recall/ num:.4f},F1: {f1/ num:.4f}, Num:{num}")

    total_acc = sum(acc_list) / sum(num_list)
    total_hit = sum(hit_list) / sum(num_list)
    total_f1 = sum(f1_list) / sum(num_list)
    total_pre = sum(pre_list) / sum(num_list)
    total_recall = sum(recall_list) / sum(num_list)
    logger.warning(f"All: Accuracy: {total_acc:.4f}, Hit: {total_hit:.4f}, Precision: {total_pre:.4f}, Recall: {total_recall:.4f},F1: {total_f1:.4f},total_num: {sum(num_list)}")
    

if __name__ == "__main__":
    main()




