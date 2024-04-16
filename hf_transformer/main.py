import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import datetime
import time
 
def load_model(attention: str = "sdpa"):
    model_id = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    torch_dtype=torch.float16, 
    # attn_implementation=attention,
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device="cuda",
        tokenizer=tokenizer,
    )
    return (pipeline, tokenizer)

def inference(input_query: str, pipeline: TextGenerationPipeline, tokenizer, max_length: int=30, attention: str="sdpa"):
    if attention == "flash_attention_2":
        enable_flash = True
        enable_math = False
        enable_mem_efficient = False
    else:
        enable_flash = False
        enable_math = True
        enable_mem_efficient = False
    with torch.backends.cuda.sdp_kernel(enable_flash=enable_flash, enable_mem_efficient=enable_mem_efficient, enable_math=enable_math):
        sequences = pipeline(
            input_query,
            do_sample=False,
            top_p = 0,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
        )
    return sequences

def run(attention: str = "sdpa", print_output: bool = True, max_length: int = 30):
    input_query = [
        'Can you explain how transformer models like GPT-3 work, focusing on the role of self-attention mechanisms in processing input data in concise within 100 words?',
        'Introduce yourself briefly',
        'What is the capital of France?'
    ]
    input_query = 'Can you explain how transformer models like GPT-3 work, focusing on the role of self-attention mechanisms?'

    pipeline, tokenizer = load_model(attention=attention)
    start = time.time.perf_counter()
    sequences = inference(input_query, pipeline, tokenizer, max_length, attention)
    end = time.perf_counter()
    # if print_output:
    #     print(f"########### {len(sequences)} Result #############")
    #     for seq in sequences:
    #         print(f"Result: {seq['generated_text']}")
    #     print("###############################")
    return end - start

def benchmark_hf(attention: str, offset, step):
    torch.cuda.empty_cache()
    input_query = 'What is the capital of France?'
    pipeline, tokenizer = load_model(attention=attention)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    open("benchmark.txt", "a").write(f"attention, length, memory, time\n")
    for i in range(step):
        length = offset * (i+1)
        start = time.perf_counter()
        inference(input_query, pipeline, tokenizer, length)
        end = time.perf_counter()
        print("time:", (end-start))
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
        torch.cuda.empty_cache()
        open("benchmark.txt", "a").write(f"{attention}, {length}, {mem}, {end-start}\n")
        print(f"attention: {attention} / max memory: {mem}GB / length: {length} / time: {end-start}")
    del pipeline
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"], help="Attention implementation")
    parser.add_argument("--output", type=bool, default=True, choices=[True, False], help="Print output")
    parser.add_argument("--max_length", type=int, default=30, help="Max length of generated text")
    parser.add_argument("--benchmark", type=int, default=True, help="Do benchmark")
    parser.add_argument("--bench-step", type=int, default=20, )
    parser.add_argument("--bench-offset", type=int, default=200, )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_query = 'Can you explain how transformer models like GPT-3 work, focusing on the role of self-attention mechanisms?'
    
    if args.benchmark:
        # N seq-length / 1 req fix / time + mem fig * 2
        benchmark_hf("sdpa", args.bench_offset, args.bench_step)
        benchmark_hf("flash_attention_2", args.bench_offset, args.bench_step)
    else:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        pipeline, tokenizer = load_model(attention=args.attention)
        start = time.perf_counter()
        sequences = inference(input_query, pipeline, tokenizer, args.max_length)
        end = time.perf_counter()
        print("time:", (end-start))
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
        torch.cuda.empty_cache()
        open("benchmark.txt", "a").write(f"attention: {args.attention} : max_memory: {mem}GB / out_length: {args.max_length}\n")
        print(f"attention: {args.attention} / max memory: {mem}GB / length: {args.max_length}")
