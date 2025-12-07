## <span style="font-size: 40px;">Fine-Tuning Llama 3.2 (1B & 3B) with LoRA + GGUF Deployment</span>


## Overview

This project implements parameter-efficient fine-tuning (PEFT) using LoRA on Llama-3.2 models, converts the fine-tuned weights into GGUF format for CPU-only environments, and deploys a Gradio UI on HuggingFace Spaces.

In part 2, we compare the inference speed of 1B and 3B models under limited compute (batch size = 2, 500 steps) and evaluate the fine-tuned 1B model quality using an automatic judge model.

# Part 1 — Fine-Tuning & UI Deployment
## 1. Training with Checkpoints Saved.

TrainingArguments with checkpointing

```python
  args = TrainingArguments(
      ...
      max_steps = 500,
      save_steps = 50,
      save_total_limit = 3,
      output_dir = "/content/drive/MyDrive/LLM_fineTuning/outputs_1B_train",
  ),
```
To resume training after a disconnect:
```python
trainer.train(resume_from_checkpoint=True)
```

## 2. Converting the LoRA-Fine-Tuned Model to GGUF (q4_k_m) for CPU Inference

GGUF export code
```python
!python llama.cpp/convert_hf_to_gguf.py \
    /content/drive/MyDrive/LLM_fineTuning/model_1B_train \
    --outfile temp_fp16.gguf \
    --outtype f16

!python llama.cpp/build/bin/llama-quantize.py \
  /content/temp_fp16.gguf \
  --outfile model_1B_train_q4_k_m.gguf" \
  --outtype q4_k_m
```

## 3. Deploying a Gradio UI on HuggingFace Spaces

The chatbot_based UI features included: system role, conversation setting up, message box and inference speed test.

👉 [The chatbot we created](https://huggingface.co/spaces/Alyssa12375/IrisPlus)

# Part 2 — Model Performance Evalution and Comparison

## 1. Inference Speed Comparison (CPU-only)

Inference Speed Results of Fine-tuned 1B and 3B LLMs with prompt "Calculate the length of the hypotenuse of a right triangle with right angle":

| Model | Format | Total Time (s) | Tokens/sec | Tokens |
|-------|--------|----------------|------------|----------|
| Llama-3.2-1B-Instruct | Q4_K_M | 12.340 | 16.208 | 200 |
| Llama-3.2-3B-Instruct | Q4_K_M | 25.512 | 7.840 | 200 |

Summary：
1B is significantly faster on CPU and ideal for deployment.

## 2. Model Quality Evaluation Using Qwen2.5-7B Judge

We sampled 30 prompts from the evaluation set and scored base vs fine-tuned outputs using: Qwen2.5-7B-Instruct-bnb-4bit as the judge model.

```python
judge_prompt = f"""
            "role": "system",
            "content": (
                "You are a strict judge.\n"
                "You must choose which answer is better.\n"
                "Reply with ONLY ONE letter:\n"
                "A (Answer A), B (Answer B), or T (Tie).\n"
                "Do not explain."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Instruction:\n{user_prompt}\n\n"
                f"Answer A:\n{A}\n\n"
                f"Answer B:\n{B}\n\n"
                "Which answer is better?\n"
                "Reply with ONLY ONE letter."
            )
"""
```
The results(prompt, outputs of both models and winners for each question) are shown in eval_results_unsloth_only(1).json.

The win rate above 60% indicates meaningful improvement. But it does not mean the winner gives really meaninggul answers.
