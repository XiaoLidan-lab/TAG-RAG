import json
import re

def txt_to_json(txt_file, json_file):
    data = []
    i = 1001
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue  # 跳过格式错误的行
            
            question = parts[0]
            answer = parts[1].split("|")  # 将答案拆分成列表
            q_entity = re.findall(r"\[(.*?)\]", question)
            question_cleaned = re.sub(r"\[|\]", "", question)  # 仅去掉方括号，保留内容
            
            data.append({
                "id": f"{i}",
                "question": question_cleaned,
                "answer": answer,
                "q_entity": q_entity,
                "a_entity": answer
            })
            i += 1
            if i >= 1401:
                break
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 示例用法
txt_to_json("data/MetaQA/qa_dev.txt", "data/MetaQA/valid.json")
