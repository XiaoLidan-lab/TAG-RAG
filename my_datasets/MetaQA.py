import torch
import json
import logging
from collections import defaultdict
from text_emb_model import BaseTextEmbModel
from torch_geometric.data import Data, InMemoryDataset
from ultra.tasks import build_relation_graph
import datasets
from omegaconf import DictConfig
import os
from torch.utils import data as torch_data
from torch.nn.functional import cosine_similarity

logger = logging.getLogger(__name__)


class metaqa(InMemoryDataset):
    def __init__(
        self,
        root: str,
        data_name: str,
    ) -> None:
        self.data_name = data_name
        super().__init__(root, None, None)
        self.kg_data, self.silces = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]), weights_only=False)
        self.rel_emb_dim = self.kg_data.rel_emb.shape[-1]
        print(os.path.join(self.processed_dir, self.processed_file_names[1]))
        self.qa_data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[1]), weights_only=False)
        # self.process()
        self.load_property()
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(
            str(self.root), 
            str(self.data_name), 
            "raw"
        )
    
    @property
    def raw_file_names(self) -> str:
        return ["kg.txt", "train.json", "test.json", "valid.json"]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(
            str(self.root),
            str(self.data_name),
            "processed"
        )
    @property
    def processed_file_names(self) -> str:
        return ["kg_data.pt", "qa_data.pt"]
    
        
    def load_file(
        self, triplet_file: str, inv_entity_vocab: dict, inv_rel_vocab: dict
    ) -> dict:
        """Load a knowledge graph file and return the processed data."""

        triplets = []  # Triples with inverse relations
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    u, r, v = line.strip().split("|")
                except Exception as e:
                    logger.error(f"Error in line: {line}, {e}, Skipping")
                    continue
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_nodes": len(inv_entity_vocab),  # entity_cnt,
            "num_relations": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }
            
    def load_property(self) -> None:
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
            
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)
            
        if os.path.exists(os.path.join(self.raw_dir, "train.json")):
            with open(os.path.join(self.raw_dir, "train.json")) as fin:
                self.raw_train_data = json.load(fin)
        else:
            self.raw_train_data = []

        if os.path.exists(os.path.join(self.raw_dir, "test.json")):
            with open(os.path.join(self.raw_dir, "test.json")) as fin:
                self.raw_test_data = json.load(fin)
        else:
            self.raw_test_data = []
        
        if os.path.exists(os.path.join(self.raw_dir, "valid.json")):
            with open(os.path.join(self.raw_dir, "valid.json")) as fin:
                self.raw_valid_data = json.load(fin)
        else:
            self.raw_valid_data = []
            
        
        self.ent_emb = torch.load(os.path.join(self.processed_dir, "ent_emb.pt"), weights_only=False)

    # kg data
    
    def process(self) -> None:
        
        if not os.path.exists(os.path.join(self.processed_dir, "ent2id.json")) or not os.path.exists(os.path.join(self.processed_dir, "rel2id.json")):
            # kg_result = process.load_from_list(data_dic["graph"], inv_entity_vocab={}, inv_rel_vocab={})
            kg_result = self.load_file(os.path.join(self.raw_dir, self.raw_file_names[0]), inv_entity_vocab={}, inv_rel_vocab={})

            num_nodes = kg_result["num_nodes"]
            num_relations = kg_result["num_relations"]
            kg_triplets = kg_result["triplets"]

            ent2id, rel2id = kg_result["inv_entity_vocab"], kg_result["inv_rel_vocab"]
            # add inverse edges
            train_target_edges = torch.tensor(
                [[t[0], t[1]] for t in kg_triplets], dtype = torch.long
            ).t()  # shape:[2, N]  edge:(head, tail)
            train_target_etypes = torch.tensor([t[2] for t in kg_triplets])
            train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim = 1) # shape[2, 2N]
            train_etypes = torch.cat(
                [train_target_etypes, train_target_etypes + num_relations] # etype of inverse edge
            )
            id2rel = {v: k for k, v in rel2id.items()}
            for etype in train_etypes:
                if etype.item() >= num_relations:
                    raw_etype = etype - num_relations
                    raw_rel = id2rel[raw_etype.item()]
                    rel2id["inverse_" + raw_rel] = etype.item()
                    
            with open(self.processed_dir + "/ent2id.json", "w") as f:
                json.dump(ent2id, f)
                
            with open(self.processed_dir + "/rel2id.json", "w") as f:
                json.dump(rel2id, f)
            
            
        # entity embedding 
        if not os.path.exists(os.path.join(self.processed_dir, "ent_emb.pt")):
            logger.info("entity embedding……")
            text_emb_model = BaseTextEmbModel("sentence-transformers/all-mpnet-base-v2", False, 32)
            ent_emb = text_emb_model.encode(list(ent2id.keys()), is_query=False).cpu()
            torch.save(ent_emb, os.path.join(self.processed_dir, "ent_emb.pt"))
            self.ent_emb = torch.load(os.path.join(self.processed_dir, "ent_emb.pt"), weights_only=False)

        # relation embedding
        if not os.path.exists(os.path.join(self.processed_dir, self.processed_file_names[0])):
            logger.info("relation embedding……")
            text_emb_model = BaseTextEmbModel("sentence-transformers/all-mpnet-base-v2", False, 32)
            rel_emb = text_emb_model.encode(list(rel2id.keys()), is_query=False).cpu()
            kg_data = Data(
                edge_index = train_edges,
                edge_type = train_etypes,
                num_nodes = num_nodes,
                target_egde_index = train_target_edges,
                target_egde_type = train_target_etypes,
                num_relations = num_relations * 2,
                rel_emb = rel_emb
            )

            # build graph of relations 
            kg_data = build_relation_graph(kg_data)

            torch.save((self.collate([kg_data])), os.path.join(self.processed_dir, self.processed_file_names[0]))


        # qa data
        if not os.path.exists(os.path.join(self.processed_dir, self.processed_file_names[1])):
            text_emb_model = BaseTextEmbModel("sentence-transformers/all-mpnet-base-v2", False, 32)
            sample_id = []
            questions = []
            question_entities_masks = []
            answer_entities_masks = []
            num_samples = []

            file_names = self.raw_file_names[1: 4]
            for file in file_names:
                path = os.path.join(self.raw_dir, file)
                if not os.path.exists(path):
                    num_samples.append(0)
                    continue 
                num_sample = 0
                with open(path) as fin:
                    data = json.load(fin)
                    for index, item in enumerate(data):
                        question_entities = [
                            ent2id[x]
                            for x in item["q_entity"]
                            if x in ent2id
                        ]
                        
                        answer_entities = [
                            ent2id[x]
                            for x in item["a_entity"]
                            if x in ent2id
                        ]
                        # if file_names == self.raw_file_names[1] or file_names == self.raw_file_names[3]:
                        # # Skip samples if any of the entities or documens are empty
                        #     if any(
                        #         len(x) == 0
                        #         for x in [
                        #             question_entities,
                        #             answer_entities
                        #         ]
                        #     ):  
                        #         self.ignore_index.append(index)
                        #         continue
                        if any(
                            len(x) == 0
                            for x in [
                                answer_entities
                            ]
                        ):  # answer embedding
                            a_emb = text_emb_model.encode(item["a_entity"], is_query = False).cpu()
                            for a in a_emb:
                                cos_sim = cosine_similarity(a.unsqueeze(0), self.ent_emb)
                                _, answer_entities = torch.topk(cos_sim, k=5)
            
                        if any(
                            len(x) == 0
                            for x in [
                                question_entities
                            ]
                        ): 
                            logger.warning("Question entities not Found")      
                            q_emb = text_emb_model.encode(item["q_entity"], is_query = False).cpu()
                            for q in q_emb:
                                cos_sim = cosine_similarity(q.unsqueeze(0), self.ent_emb)
                                _, question_entities = torch.topk(cos_sim, k=1)
                                
                        num_sample += 1
                        sample_id.append(index)
                        question = item["question"]
                        questions.append(question)
                        
                        question_mask = torch.zeros(num_nodes)
                        question_mask[question_entities] = 1
                        question_entities_masks.append(question_mask)
                        answer_mask = torch.zeros(num_nodes)
                        answer_mask[answer_entities] = 1
                        answer_entities_masks.append(answer_mask)
                        
                    num_samples.append(num_sample)
            # question embedding
            logger.info("question embedding……")
            question_embeddings = text_emb_model.encode(
                questions,
                is_query = True,
            ).cpu()
            question_entities_masks = torch.stack(question_entities_masks)
            answer_entities_masks = torch.stack(answer_entities_masks)
            sample_id = torch.tensor(sample_id, dtype = torch.long)

            dataset = datasets.Dataset.from_dict(
                {
                    "question_embeddings": question_embeddings,
                    "question_entities_masks": question_entities_masks,
                    "answer_entities_masks": answer_entities_masks,
                    "sample_id": sample_id,
                }
            ).with_format("torch")
            
            offset = 0
            splits = []
            for num_sample in num_samples:
                split = torch_data.Subset(dataset, range(offset, offset + num_sample))
                splits.append(split)
                offset += num_sample
            torch.save(splits, os.path.join(self.processed_dir, "qa_data.pt"))



