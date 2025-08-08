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


```python
from transformers import AutoTokenizer, GPT2LMHeadModel

model_name = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

```




    Embedding(51201, 768)




```python
from datasets import Dataset

url = "https://ko.wikipedia.org/wiki/인공지능"
text_data = get_text_from_korean_url(url)

# Dataset 형태로 구성
dataset = Dataset.from_dict({"text": [text_data]})
```


```python
def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
```


    Map:   0%|          | 0/1 [00:00<?, ? examples/s]



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

    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
    



    
      <progress value='3' max='3' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [3/3 00:06, Epoch 3/3]
    




    TrainOutput(global_step=3, training_loss=4.529757817586263, metrics={'train_runtime': 7.5884, 'train_samples_per_second': 0.395, 'train_steps_per_second': 0.395, 'total_flos': 783876096000.0, 'train_loss': 4.529757817586263, 'epoch': 3.0})




```python
model.eval()

prompt = "인공지능이란"  # 시작 문장
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)
input_ids = input_ids.to(device) 

generated_ids = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=5,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

print("\n📌 생성된 문장 5개:")
for i, gen_id in enumerate(generated_ids):
    text = tokenizer.decode(gen_id, skip_special_tokens=True)
    print(f"{i+1}. {text}\n")
```

    
    📌 생성된 문장 5개:
    1. 인공지능이란 인공지능에 대해 어느 정도 이해했는지, 아니면 인공지능에 대한 이해 수준이 어느 정도인지 파악하기 위해 연구자들을 찾는 것 역시 쉬운 일이 아니다.
    다만 이러한 일은 연구자들이 연구 활동을 시작할 때 반드시 해야 하는 일이므로,
    
    2. 인공지능이란 무엇인가?
    아마 프로그래밍 언어학에서 프로그래밍 언어학과는 그 정의적인 부분, 즉 모든 사용자에 대한 컴퓨터적 이해와 관련된 가장 강력한 개념 중 하나로 인식되고 있다.
    프로그래밍 언어를 연구하는 사람들은 그 개념들을 어떻게
    
    3. 인공지능이란게 뭘까요?
    그저 말하려는 그 이유와 답은 니다.
    사실 이라는 이 인공지능은 일종의 마이너리티를 다루었고 이라는 개념을 완전히 무시했다는 거죠.
    그래서 마
    
    4. 인공지능이란 말이다. 지난 2000년 초순부터 이달 초까지 미국 실리콘밸리의 IT기업 최고경영자(CEO)들이 가장 선호하는 투자항목의 하나로 꼽혔던 것이 바로 '기술'과 '기업'이었다.
    지난 2004년의
    
    5. 인공지능이란 어떤 것인지, 그리고 그게 어떤 것인지 등등은 여전히 수수께끼로 남아 있다.
    그러나 우리가 사용하는 어떤 다른 도구들은 어떤 것들이든지 간에, 그걸 알고 있는 한 그들은 아주 오래전부터 그런 것이 가능할 것이라는 생각을 해왔다.
    어떤 사람들은 자신이 한
    
    
