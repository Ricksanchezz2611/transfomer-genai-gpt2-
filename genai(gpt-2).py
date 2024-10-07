import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt,len,max_length=50):

  input_ids = tokenizer.encode(prompt, return_tensors='pt', padding='max_length', max_length=len)

    # Create attention mask
  attention_mask = (input_ids != pad_token_id).long()

  with torch.no_grad():
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=pad_token_id,
        no_repeat_ngram_size=2,
        temperature=0.4,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

  generated_text = tokenizer.decode(output[0],skip_special_tokens=True)
  return generated_text

prompt = "the spiderman"
prompt_len=len(prompt)
generated = generate_text(prompt,prompt_len)
print(generated)