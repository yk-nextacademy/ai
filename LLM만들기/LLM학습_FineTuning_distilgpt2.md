# LLM 파인튜닝

본 코드는 Hugging Face의 `transformers`와 `datasets` 라이브러리를 사용하여 한국어 GPT-2 모델(`KoGPT2`)을 간단한 질의응답 데이터로 파인튜닝하는 과정을 보여주고 있습니다.

## 라이브러리 로드


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
```

## 간단한 한글 질문-답변 toy 데이터셋 준비

- 아주 간단한 질의응답 형태의 딕셔너리 리스트를 생성
- 'prompt'가 질문, 'completion'이 답변 역할


```python
data = [
    {"prompt": "안녕하세요?", "completion": " 안녕하세요! 무엇을 도와드릴까요?"},
    {"prompt": "당신의 이름은?", "completion": " 저는 토이 모델입니다."},
    {"prompt": "AI란 무엇인가요?", "completion": " 인공지능을 뜻합니다."},
    {"prompt": "재미있는 농담 말해줘.", "completion": " 닭이 길을 건넌 이유는 다른 쪽에 가기 위해서입니다!"},
]
```

## 데이터 전처리

- preprocess() 정의
    - 모델 학습에 사용할 "질문: ...\n답변: ..." 형식의 텍스트를 생성
- 파이썬 리스트를 Hugging Face의 Dataset 객체로 변환
    - Dataset의 각 항목에 대해 위에서 정의한 preprocess 함수를 적용


```python
def preprocess(example):
    text = f"질문: {example['prompt']}\n답변:{example['completion']}"
    return {"text": text}

dataset = Dataset.from_list(data)
dataset = dataset.map(preprocess)
print(dataset[0])
```


    Map:   0%|          | 0/4 [00:00<?, ? examples/s]


    {'prompt': '안녕하세요?', 'completion': ' 안녕하세요! 무엇을 도와드릴까요?', 'text': '질문: 안녕하세요?\n답변: 안녕하세요! 무엇을 도와드릴까요?'}
    

## 모델과 토크나이저 로드
- 모델
    - skt/kogpt2-base-v2
    - SKT에서 공개한 KoGPT2
- tokenizer.pad_token_id 설정
    - None -> tokenizer.eos_token_id-1


```python
model_name = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# KoGPT2 토크나이저의 특별 토큰 ID를 설정
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
```

## 토크나이징 함수
- 토크나이징을 수행하는 함수입니다.
- `padding="max_length"`: 모든 문장의 길이를 max_length에 맞춰 패딩
- `truncation=True`: 문장 길이가 max_length보다 길면 잘라 냄
- `max_length=64`: 최대 문장 길이를 64로 설정
- `return_attention_mask=True`: 패딩된 토큰을 무시하도록 attention_mask를 생성


```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_attention_mask=True
    )
```

## 토크나이징

- 손실(loss) 계산 시 패딩 토큰을 무시하기 위해 레이블을 마스킹


```python
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

print(tokenized_dataset[0])
```


    Map:   0%|          | 0/4 [00:00<?, ? examples/s]


    {'prompt': '안녕하세요?', 'completion': ' 안녕하세요! 무엇을 도와드릴까요?', 'input_ids': [24454, 401, 25906, 8702, 7801, 13675, 7192, 7643, 401, 25906, 8702, 7801, 8084, 376, 22375, 14355, 7281, 7481, 6969, 8084, 406, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    


```python
def mask_labels(examples):
    labels = []
    for input_ids in examples['input_ids']:
        label = [tok if tok != tokenizer.pad_token_id else -100 for tok in input_ids]
        labels.append(label)
    return {"labels": labels}
    
tokenized_dataset = tokenized_dataset.map(mask_labels, batched=True)
print(tokenized_dataset[0])
```


    Map:   0%|          | 0/4 [00:00<?, ? examples/s]


    {'prompt': '안녕하세요?', 'completion': ' 안녕하세요! 무엇을 도와드릴까요?', 'input_ids': [24454, 401, 25906, 8702, 7801, 13675, 7192, 7643, 401, 25906, 8702, 7801, 8084, 376, 22375, 14355, 7281, 7481, 6969, 8084, 406, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': [24454, 401, 25906, 8702, 7801, 13675, 7192, 7643, 401, 25906, 8702, 7801, 8084, 376, 22375, 14355, 7281, 7481, 6969, 8084, 406, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]}
    

## 레이블 설정 (causal LM용)

- Causal LM은 자기회귀(autoregressive) 모델이므로, `input_ids` 자체가 `labels`가 됨


```python
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})
print(tokenized_dataset[0])
```


    Map:   0%|          | 0/4 [00:00<?, ? examples/s]


    {'prompt': '안녕하세요?', 'completion': ' 안녕하세요! 무엇을 도와드릴까요?', 'input_ids': [24454, 401, 25906, 8702, 7801, 13675, 7192, 7643, 401, 25906, 8702, 7801, 8084, 376, 22375, 14355, 7281, 7481, 6969, 8084, 406, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': [24454, 401, 25906, 8702, 7801, 13675, 7192, 7643, 401, 25906, 8702, 7801, 8084, 376, 22375, 14355, 7281, 7481, 6969, 8084, 406, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201, 51201]}
    

## Trainer 세팅

- output_dir="./ko-finetune", # 학습 결과(모델, 체크포인트)가 저장될 디렉토리
- per_device_train_batch_size=4, # 각 장치(GPU 또는 CPU)당 배치 크기
- num_train_epochs=10, # 전체 데이터셋을 10번 반복 학습합니다.
- logging_steps=1, # 1 스텝마다 로그를 출력합니다.
- learning_rate=3e-5, # 학습률 설정
- weight_decay=0.01, # 가중치 감소(Weight Decay) 설정
- use_cpu=True, # CPU를 사용하도록 설정합니다. (GPU가 없을 때)
- logging_dir='./logs', # 로그가 저장될 디렉토리


```python
training_args = TrainingArguments(
    output_dir="./ko-finetune",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    logging_steps=10,
    learning_rate=3e-5,
    weight_decay=0.01,
    use_cpu=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer 
)
```

## 답변 생성 함수

- inputs["input_ids"],
- attention_mask=inputs["attention_mask"],
- max_length=64, # 최대 생성 길이
- do_sample=True, # 확률적 샘플링을 사용하여 더 다양한 답변을 생성
- temperature=0.3, # 낮을수록 보수적, 높을수록 창의적인 답변 생성
- top_p=0.9, # 상위 p% 확률의 토큰들 중에서만 샘플링
- repetition_penalty=1.2, # 반복을 줄이는 페널티
- no_repeat_ngram_size=2 # n-gram 반복을 방지


```python
def generate_answer(model, tokenizer, prompt):
    model.eval() # 모델을 평가 모드로 전환
    
    inputs = tokenizer(
        f"질문: {prompt}\n답변:",
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=64,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        # 기울기 계산을 비활성화하여 메모리를 절약하고 속도를 높입니다.
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64, # 최대 생성 길이
            do_sample=True, # 확률적 샘플링을 사용하여 더 다양한 답변을 생성
            temperature=0.3, # 낮을수록 보수적, 높을수록 창의적인 답변 생성
            top_p=0.9, # 상위 p% 확률의 토큰들 중에서만 샘플링
            repetition_penalty=1.2, # 반복을 줄이는 페널티
            no_repeat_ngram_size=2 # n-gram 반복을 방지
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "답변:" in decoded:
        answer = decoded.split("답변:")[-1].strip()
    else:
        answer = decoded.strip()

    return answer
```

## 파인튜닝 전 출력


```python
print("=== 파인튜닝 전 모델 답변 ===")
for item in data:
    print(f"질문: {item['prompt']}")
    print(f"답변: {generate_answer(model, tokenizer, item['prompt'])}")
    print("-" * 30)
```

    === 파인튜닝 전 모델 답변 ===
    질문: 안녕하세요?
    답변: 저는 지난번에도 말씀드렸듯이 제가 이번에 또 다른 분이 계신데요.
    저는 이번에도 그분에게 부탁을 드리고 싶어요.
    제가 지금 말씀드리려고 하는 것은 저희들이 그동안에 여러 가지 어려운 여건 속에서도 열심히 노력해 주셨기
    ------------------------------
    질문: 당신의 이름은?
    답변: 당신은 지금 이 순간에도, 그리고 앞으로도 계속될 것입니다.
    당신의 이름을 기억하세요.
    그렇지 않으면 당신의 이름이 뭔지도 잊어버리고 말 겁니다.
    이제부터 당신의 이름 앞에 서게 될 거라고 생각합니다.
    당신이 누군지, 어떤 사람이 누구인지, 무슨 말을
    ------------------------------
    질문: AI란 무엇인가요?
    답변: IBM은 ‘AI를 통해 인간의 삶을 더 나은 방향으로 바꿀 수 있다’고 말했습니다.
    이 말은 인간이 가진 모든 것을 가지고 있는 것이 아니라 인간 스스로 만들어가는 것이라는 뜻입니다.
    그런데 AI는 인간을 어떻게 변화시킬까요?
    답안: “인간에게 필요한
    ------------------------------
    질문: 재미있는 농담 말해줘.
    답변: 이랑 같이 놀아줄래?
    ( 웃음 ) : 아, 이거 진짜 웃긴데? ( 웃음 )
    여기서 말하는 '웃음'은 '너무' 또는 '즐겁다'는 뜻인데, 이는 '내가 너무 즐거워'라는 의미
    ------------------------------
    

## 파인튜닝 수행


```python
trainer.train()
```

    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.  

    TrainOutput(global_step=20, training_loss=1.1369717210531234, metrics={'train_runtime': 22.8282, 'train_samples_per_second': 3.504, 'train_steps_per_second': 0.876, 'total_flos': 2612920320000.0, 'train_loss': 1.1369717210531234, 'epoch': 20.0})



## 파인튜닝 후 출력


```python
print("\n=== 파인튜닝 후 모델 답변 ===")
for item in data:
    print(f"질문: {item['prompt']}")
    print(f"답변: {generate_answer(model, tokenizer, item['prompt'])}")
    print("-" * 30)
```

    
    === 파인튜닝 후 모델 답변 ===
    질문: 안녕하세요?
    답변: 저는 토이 모델입니다.말이나말로말을 대신합니다.말과언어도말,말의 뜻을 같이합니다!말은 무엇을 뜻입니까?언어도말이케이션담나눔 말입니다.어떤 뜻으로 길을
    ------------------------------
    질문: 당신의 이름은?
    답변: 저는 토이 모델입니다. 대물림한 경우처럼 대신울 수 있습니다.말고도 다른 쪽에 가는 길이 있을까요? 말이나 뜻을 같이하면 됩니다!을이의 뜻도 이와 같습니다.이 뜻이 같은 것은 무엇이
    ------------------------------
    질문: AI란 무엇인가요?
    답변: 인공지능을 뜻합니다.이 모델입니다.을이에요.말도담 이기도 합니다.과에세이 이와 같습니다.이라는 도구를 이용해 길을 건넌 이유는 다른 쪽에 가기 위해서입니다!은라는 도구를 이용해서 길을
    ------------------------------
    질문: 재미있는 농담 말해줘.
    답변: 닭이 길을 건넌 이유는 다른 쪽에 가기 위해서입니다! 이기는 길입니다.이 모델은말고도 다른 쪽에도 가려고 합니다.을이는 무엇을 도와드릴까요?라는 도구를 이용해과 뜻을 같이합니다.
    ------------------------------
    
