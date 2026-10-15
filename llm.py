import torch
import contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

#<s>[INST] 你好，帮我总结一下这篇文章？ [/INST] 当然！这篇文章的主要内容是... </s>
BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100

class LLM(torch.nn.Module):
    
    def __init__(
        self,
        llm_model_path,
        llm_frozen,
        device,
    ):
        super().__init__()
        self.max_txt_len = 512
        # self.max_new_tokens = 32
        self.max_new_tokens = 64
        self._device = device
        
        print("Loading LLAMA with QLoRA...")

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, use_fast=False)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        ).to(self._device)

        if llm_frozen:
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with QLoRA")
            model = prepare_model_for_kbit_training(model)
            
            lora_r = 16
            lora_alpha = 64
            lora_dropout = 0.1
            lora_target_modules = ['gate_proj', 'k_proj', 'v_proj', 'q_proj', 'up_proj', 'down_proj', 'o_proj']
            # lora_target_modules = [
            #     "q_proj",
            #     "v_proj",
            # ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            
        self.model = model
        print("Finish loading LLAMA with QLoRA!")

        self.word_embedding = self.model.model.get_input_embeddings().to(self._device)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

        
    def forward(self, samples):
        # -> { 'input_ids': [...], 'attention_mask': [...] }
        questions = self.tokenizer(samples["question"], add_special_tokens = False)
        retrieved_ent = [self.tokenizer(ent, add_special_tokens=False) for ent in samples["ent"]]
        answer = self.tokenizer(samples["answer"], add_special_tokens = False)
        
        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self._device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id, device = self._device)).unsqueeze(0)
        
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_answer_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token 
            ent_input_ids = []
            for ent in retrieved_ent[i].input_ids:
                ent_input_ids = ent_input_ids + ent[:self.max_txt_len]
            answer_input_ids = answer.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = ent_input_ids + questions.input_ids[i] + eos_user_tokens.input_ids + answer_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=self._device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim = 0)
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            answer_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(answer_input_ids)) + answer_input_ids
            batch_answer_input_ids.append(answer_input_ids)
            
        # Pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
            batch_answer_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_answer_input_ids[i]
            
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self._device)
        attention_mask = torch.tensor(batch_attention_mask, device=self._device)
        answer_input_ids = torch.tensor(batch_answer_input_ids, device=self._device)   
            
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=answer_input_ids,
            )
            

        return outputs.loss   
            
    def inference(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        # retrieved_ent = [self.tokenizer(ent, add_special_tokens=False) for ent in samples["inference_path"]]
        retrieved_ent = [self.tokenizer(ent, add_special_tokens=False) for ent in samples["ent"]]
        # prompt1 = "Answer the question based on the relevant entities below: "
        # prompt1 = self.tokenizer(prompt1, add_special_tokens=False)
        # prompt2 = "The answer may be not only one, and you can consider the information as follows. If you think the answers are as follows, just choose them to answer my question concisely. If not, answer the question according to your own knowledge:"
        # prompt2 = self.tokenizer(prompt2, add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self._device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id, device=self._device)).unsqueeze(0)
        
            
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        
        for i in range(batch_size):
            ent_input_ids = []
            for ent in retrieved_ent[i].input_ids:
                ent_input_ids = ent_input_ids + ent[:self.max_txt_len]
            
            # input_ids = questions.input_ids[i] + eos_user_tokens.input_ids
            input_ids = ent_input_ids + questions.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            # input_ids = retrieved_ent.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self._device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self._device)
        attention_mask = torch.tensor(batch_attention_mask, device=self._device)
        
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return {'id': samples['id'],
                'pred': pred,
                'answer': samples['answer'],
                'question': samples['question'],
                'ent': samples['ent']}


        
        
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

