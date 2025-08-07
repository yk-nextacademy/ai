## 데이터 준비  

```python
import requests
from bs4 import BeautifulSoup
import re

def get_text_from_korean_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    # 너무 짧은 문장, 특수문자 제거
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^가-힣0-9a-zA-Z .,?!]', '', text)
    return text
```

## 토큰화  

```python
from transformers import AutoTokenizer, GPT2LMHeadModel

model_name = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
```

## 데이터 로더

```python
from datasets import Dataset

url = "https://ko.wikipedia.org/wiki/인공지능"
text_data = get_text_from_korean_url(url)

# Dataset 형태로 구성
dataset = Dataset.from_dict({"text": [text_data]})

def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
```

## 학습

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./Result02",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

## 결과 확인  
```python
model.eval()

prompt = "강한 인공지능이란"  # 시작 문장
input_ids = tokenizer.encode(prompt, return_tensors="pt")

generated_ids = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=5,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

print("\n생성된 문장 5개:")
for i, gen_id in enumerate(generated_ids):
    text = tokenizer.decode(gen_id, skip_special_tokens=True)
    print(f"{i+1}. {text}\n")
```
