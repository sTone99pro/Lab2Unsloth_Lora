# Lab2-Unsloth_Lora_fine_tune

## This project fine-tunes unsloth/Llama-3.2-1B-Instruct-bnb-4bit using the mlabonne/FineTome-100k dataset on Google Colab.
Checkpoints are saved to Google Drive during training, and the final model is merged and converted to GGUF (q8_0) for inference with llama.cpp.

### 1.Checkpoint Saving
drive.mount('/content/drive')
output_dir = "/content/drive/MyDrive/llm_checkpoints"
save_steps = 50,          # save every 50 steps
save_total_limit = 2,     # keep only 2 checkpoints

### 2.Merge LoRA Weights


BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
LORA_PATH = "Path"


model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="mps",
    dtype="auto"
)
model = PeftModel.from_pretrained(model, LORA_PATH)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained("merged_model")

### 3.Convert to GGUF (q8_0) using llama.cpp
python convert-hf-to-gguf.py ../merged_model \
    --outfile llama-finetuned-q8_0.gguf \
    --outtype q8_0

### 4.Upload to Huggingface
