# convert_jax_to_torch.py

import sys
from transformers import T5ForConditionalGeneration

model_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else model_path


model_torch = T5ForConditionalGeneration.from_pretrained(model_path, from_flax=True)
model_torch.save_pretrained(output_path)