# LLM + RAG

- LLM Model: [kakaocorp / kanana-nano-2.1b-instruct](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct)
- Dataset  : [갤럭시 Z 폴드7 자급제 스펙](https://www.samsung.com/sec/smartphones/galaxy-z-fold7/buy/?modelCode=SM-F966NDBEKOO)

## 라이브러리 로드


```python
# pip install torch transformers

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
```

## 데이터셋 준비


```python
def read_json_to_documents(json_path, text_key="content"):
    documents = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            if isinstance(item, dict) and text_key in item:
                text = item[text_key]
                if text:
                    documents.append(text)
            elif isinstance(item, str):
                documents.append(item)
    return documents
```

## 답변 생성


```python
def generate_answer(model, tokenizer, messages, device, max_tokens=256):
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in response:
        answer = response.split("assistant", 1)[-1].strip()
    else:
        answer = response.strip()
    return answer
```

## Main 함수


```python
def main():
    json_path = 'GZFold7_Spec.json'
    text_key = "content"  # JSON 내 텍스트 키명

    print("JSON 데이터 로딩 중...")
    documents = read_json_to_documents(json_path, text_key)
    print(f"총 문서 수: {len(documents)}")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")

    model_name = "kakaocorp/kanana-nano-2.1b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    ).to(device)

    print("\n질문을 입력하세요. 종료하려면 엔터(빈 입력) 또는 'exit'를 입력하세요.")

    while True:
        query = input("\n질문: ").strip()
        if query == "" or query.lower() == "exit":
            print("종료합니다.")
            break

        # RAG 미적용 답변
        simple_messages = [
            {"role": "system", "content": "너는 삼성전자 도우미야."},
            {"role": "user", "content": query}
        ]
        simple_answer = generate_answer(model, tokenizer, simple_messages, device)

        # RAG 적용 답변 (간단 키워드 검색)
        retrieved = [doc for doc in documents if any(q in doc for q in query.split())]

        context = "\n".join(retrieved) if retrieved else "검색된 내용이 없습니다."

        rag_messages = [
            {"role": "system", "content": "너는 삼성전자 도우미야."},
            {"role": "user", "content": f"다음 문서를 참고해서 질문에 답변해줘.\n문서:\n{context}\n질문: {query}"}
        ]
        rag_answer = generate_answer(model, tokenizer, rag_messages, device)

        print("\n===== RAG 비적용 (질문만) 답변 =====")
        print(simple_answer)
        print("\n===== RAG 적용 (검색 문서 참고) 답변 =====")
        print(rag_answer)


if __name__ == "__main__":
    main()
```

    JSON 데이터 로딩 중...
    총 문서 수: 184
    
    질문을 입력하세요. 종료하려면 엔터(빈 입력) 또는 'exit'를 입력하세요.
    

    
    질문:  폴드7 무게를 알려줘.
    

    
    ===== RAG 비적용 (질문만) 답변 =====
    삼성전자 폴드7(Fold 7)의 무게는 약 400g입니다. 폴드7은 접이식 스마트폰으로, 두꺼운 접히는 부분을 고려하더라도 사용자 경험이 편리하도록 설계되었습니다. 정확한 무게는 제조사나 모델에 따라 약간의 차이가 있을 수 있지만, 일반적으로 이 범위 내에 있습니다. 추가로 궁금한 사항이 있으면 언제든지 물어보세요!
    
    ===== RAG 적용 (검색 문서 참고) 답변 =====
    폴드7의 무게는 215 그램(g)입니다.
    

    
    질문:  
    

    종료합니다.
 
