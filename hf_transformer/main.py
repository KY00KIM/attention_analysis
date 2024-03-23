import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
 
def load_model(attention: str = "sdpa"):
    model_id = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    torch_dtype=torch.bfloat16, 
    attn_implementation=attention,
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device="cuda",
        tokenizer=tokenizer,
    )
    return (pipeline, tokenizer)

def inference(input_query, pipeline, tokenizer):
    sequences = pipeline(
        input_query,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("########### Result #############")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    print("###############################")

def run(attention: str = "sdpa"):
    input_query = 'Hi I\'m a CS major college student. Introduce yourself.\n'

    pipeline, tokenizer = load_model(attention=attention)

    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
           inference(input_query, pipeline, tokenizer)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=200))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"], help="Attention implementation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(attention=args.attention)