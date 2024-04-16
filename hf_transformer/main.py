import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import datetime
 
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

def inference(input_query: str, pipeline: TextGenerationPipeline, tokenizer):
    sequences = pipeline(
        input_query,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    return sequences

def run(attention: str = "sdpa", print_output: bool = True):
    from .test_module import tmodule, input_tensor
    input_query = [
        'Can you explain how transformer models like GPT-3 work, focusing on the role of self-attention mechanisms in processing input data in concise within 100 words?',
        'Introduce yourself briefly',
        'What is the capital of France?'
    ]

    pipeline, tokenizer = load_model(attention=attention)
    cnt = 0
    # while True:
    # input_query = input("Input a prompt: (type 'end' to exit)")
    input_query = 'What is the capital of France?'
    # if input_query == "end":
    #     break

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    record_shapes=True, profile_memory=True, 
            #      schedule=torch.profiler.schedule(
            # wait=1,
            # warmup=1,
            # active=2),
                    ) as prof:
        with record_function(f"model_inference:{cnt}"):
            print("Generating...")
            sequences = inference(input_query, pipeline, tokenizer)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, ))
    current_time = datetime.datetime.now().strftime("%y%m%d-%H-%M-%S")
    prof.export_chrome_trace(f"trace{current_time}.json")

    if print_output:
        print(f"########### {len(sequences)} Result #############")
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
        print("###############################")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"], help="Attention implementation")
    parser.add_argument("--output", type=bool, default=True, choices=[True, False], help="Print output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(attention=args.attention)