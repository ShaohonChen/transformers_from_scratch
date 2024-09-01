import datasets
import transformers
import swanlab
from swanlab.integration.huggingface import SwanLabCallback
import torch.distributed as dist


def main():
    # using swanlab to save log
    swanlab.init("TransformersFromScratch")

    # load dataset
    raw_datasets = datasets.load_dataset(
        "json", data_files="data/wikipedia-zh-cn-20240820.json"
    )

    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=2333)
    print("dataset info")
    print(raw_datasets)

    # load tokenizers
    context_length = 512  # use a small context length
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

    # preprocess dataset
    def tokenize(element):
        outputs = tokenizer(
            element["completion"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    print("tokenize dataset info")
    print(tokenized_datasets)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # prepare a model from scratch
    config = transformers.AutoConfig.from_pretrained(
        "Qwen/Qwen2-0.5B",
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=12,
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = transformers.Qwen2ForCausalLM(config)
    model_size = sum(t.numel() for t in model.parameters())
    print("Model Config:")
    print(config)
    print(f"Model Size: {model_size/1000**2:.1f}M parameters")

    # train
    args = transformers.TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=24,  # 每个GPU的训练batch数
        per_device_eval_batch_size=24,  # 每个GPU的测试batch数
        eval_strategy="steps",
        eval_steps=5_000,
        logging_steps=500,
        gradient_accumulation_steps=12,  # 梯度累计总数
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        optim="adamw_torch",  # 优化器使用adamw
        lr_scheduler_type="cosine",  # 学习率衰减策略
        learning_rate=5e-4,  # 基础学习率，
        save_steps=5_000,
        save_total_limit=10,
        bf16=True,  # 开启bf16训练, 对于Amper架构以下的显卡建议替换为fp16=True
    )
    print("Train Args:")
    print(args)
    # enjoy training
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        callbacks=[SwanLabCallback()],
    )
    trainer.train()

    # save model
    model.save_pretrained("./output/")  # 保存模型的路径

    # generate
    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("GENERATE:", pipe("人工智能", num_return_sequences=1)[0]["generated_text"])
    prompts = ["牛顿是", "北京市的人口", "太平洋的面积"]
    examples = []
    for i in range(3):
        # 根据提示词生成数据
        text = pipe(prompts[i], num_return_sequences=1)[0]["generated_text"]
        text = swanlab.Text(text)
        examples.append(text)
    swanlab.log({"Generate": examples})


if __name__ == "__main__":
    main()
