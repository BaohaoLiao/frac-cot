import os
import json
import argparse

import transformers
from vllm import LLM, SamplingParams

from utils.inf import set_seed
from utils.data import load_data, parse_question, prepare_prompt
from utils.eval import extract_pred_and_parse, parse_ground_truth, obtain_nHm_scores_and_preds


def main(args):
    # Out file
    model_name = args.model_name_or_path.split("/")[-1]
    output_dir = args.output_dir
    os.makedirs(f"{output_dir}/{args.data_name}", exist_ok=True)

    out_file_prefix = f"{model_name}_seed{args.seed}_t{args.temperature}topp{args.top_p}minp{args.min_p}_" \
        f"len{args.max_tokens_per_call}_num{args.num_test_sample}_n{args.n_sampling}"
    out_file = f"{output_dir}/{args.data_name}/{out_file_prefix}H{args.num_think_chunks}m{args.num_solutions_per_chunk}.json"
    think_solutions_file = f"{output_dir}/{args.data_name}/{out_file_prefix}_think_solutions.json"
    
    # Load and prepare data
    if "math500_level" in args.data_name:
        level = int(args.data_name.strip()[-1])
        examples = load_data("math500", args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(args.data_name, args.data_dir)

    if args.num_test_sample != -1:
        examples = examples[:args.num_test_sample]

    print("=" * 50)
    print(f"{args.data_name} || #samples: {len(examples)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    samples = []
    for i, example in enumerate(examples):
        question = parse_question(example, args.data_name)
        prompt = prepare_prompt(question, tokenizer, args.data_name)
        samples.append({
            "idx": example["idx"],
            "question": question,
            "answer": example["answer"],
            "prompt": prompt,
        })
        if i == 0:
            print(prompt)

    # Load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    # Sample at the n dimension
    ## Either load existing think solutions or generate new ones
    if args.load_think_solutions:
        think_solutions_path = args.load_think_solutions
        print(f"Loading think solutions from {think_solutions_path}")
        try:
            with open(think_solutions_path, 'r') as f:
                think_solutions_data = json.load(f)
                
            loaded_think_solutions = think_solutions_data['think_solutions']
            loaded_n_sampling = think_solutions_data.get('n_sampling', 1)
            loaded_num_samples = think_solutions_data.get('num_samples', len(loaded_think_solutions) // loaded_n_sampling)

            assert loaded_n_sampling == args.n_sampling and loaded_num_samples == len(samples)
            think_solutions = loaded_think_solutions
            print(f"Successfully loaded think solutions")
        except Exception as e:
            print(f"Error loading think solutions: {e}")
            print("Falling back to generating new think solutions")
            think_solutions = None
    else:
        think_solutions = None

    if think_solutions is None:
        prompts = [sample["prompt"] for sample in samples]
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            max_tokens=args.max_tokens_per_call,
            n=args.n_sampling,
            skip_special_tokens=False,
            seed=args.seed,
        )
        llm_outputs = llm.generate(prompts, sampling_params)
        llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
        think_solutions = [
            output.text for llm_output in llm_outputs for output in llm_output.outputs
        ]  # flatten
        assert len(think_solutions) == len(prompts) * args.n_sampling
        
        if args.save_think_solutions:
            think_solutions_data = {
                'data_name': args.data_name,
                'model': args.model_name_or_path,
                'seed': args.seed,
                'temperature': args.temperature,
                'n_sampling': args.n_sampling,
                'num_samples': len(samples),
                'think_solutions': think_solutions,
            }
            try:
                with open(think_solutions_file, 'w') as f:
                    json.dump(think_solutions_data, f)
                print(f"Saved think solutions to {think_solutions_file}")
            except Exception as e:
                print(f"Error saving think solutions: {e}")

    n_solutions = []
    for think_solution in think_solutions:
        if "</think>" not in think_solution:  # solution is not generated
            n_solutions.append(think_solution)
        else:
            n_solutions.append(think_solution.split("</think>")[-1])
    
    # Vanilla sampling, i.e. H=1, m=1
    if args.num_think_chunks == 1 and args.num_solutions_per_chunk == 1:
        n_preds = [extract_pred_and_parse(sol, args.data_name) for sol in n_solutions]

        ## Organize into 3D structure (just with dimensions of [n, 1, 1])
        all_solutions = []
        all_preds = []
        for s in range(len(think_solutions)):
            sample_solutions = [[n_solutions[s]]]
            sample_preds = [[n_preds[s]]]
            all_solutions.append(sample_solutions)
            all_preds.append(sample_preds)
        
        ## Process predictions and put back into samples
        for i, sample in enumerate(samples):
            sample_solutions = []
            sample_preds = []
            for s in range(args.n_sampling):
                idx = i * args.n_sampling + s
                sample_solutions.append(all_solutions[idx])
                sample_preds.append(all_preds[idx])

            parsed_gt = parse_ground_truth(sample, args.data_name)
            new_gt, new_sample_preds, sample_scores = obtain_nHm_scores_and_preds(parsed_gt, sample_preds)
            sample.update({
                "think_solution": think_solutions[i * args.n_sampling : (i + 1) * args.n_sampling],
                "solution": sample_solutions,  # [n, 1, 1]
                "gt": new_gt,
                "pred": new_sample_preds, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "score": sample_scores, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
            })

    # H=1, m>1 (single chunk but multiple solutions)
    elif args.num_think_chunks == 1 and args.num_solutions_per_chunk > 1:
        ## Generate multiple solutions for the same thinking
        prompts = [sample["prompt"] for sample in samples]
        prompt_thinks = []
        for r, think_solution in enumerate(think_solutions):
            if "</think>" not in think_solution:
                prompt_think = prompts[r//args.n_sampling] + think_solution + "\n</think>\n\n"
            else:
                prompt_think = prompts[r//args.n_sampling] + think_solution.split("</think>")[0] + "</think>"
            
            prompt_thinks.append(prompt_think)
        
        ## Generate additional solutions for the full thinking
        solution_sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            max_tokens=args.max_tokens_per_solution,
            n=args.num_solutions_per_chunk - 1, # minus 1 because we already have one
            skip_special_tokens=False,
            seed=args.seed,
        )
        llm_outputs = llm.generate(prompt_thinks, solution_sampling_params)
        llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
        nm_solutions = [output.text for llm_output in llm_outputs for output in llm_output.outputs] # flatten
        assert len(nm_solutions) == len(prompt_thinks) * (args.num_solutions_per_chunk - 1)
        
        ## Extract predictions
        n_preds = [extract_pred_and_parse(sol, args.data_name) for sol in n_solutions]
        nm_preds = [extract_pred_and_parse(sol, args.data_name) for sol in nm_solutions]
        
        ## Organize into 3D structure (but with only one chunk dimension)
        all_solutions = []
        all_preds = []
        final_idx = 0
        for s in range(len(think_solutions)):
            ### For each sample, create one chunk with multiple solutions
            sample_solutions = [[n_solutions[s]]]
            sample_preds = [[n_preds[s]]]
            
            ### Add additional solutions at m dimension
            for a in range(args.num_solutions_per_chunk - 1):
                sample_solutions[0].append(nm_solutions[final_idx])
                sample_preds[0].append(nm_preds[final_idx])
                final_idx += 1
            
            all_solutions.append(sample_solutions)
            all_preds.append(sample_preds)
        
        ## Process results and put back into samples
        for i, sample in enumerate(samples):
            ### Get solutions and predictions for this sample
            sample_solutions = []
            sample_preds = []
            
            for s in range(args.n_sampling):
                idx = i * args.n_sampling + s
                sample_solutions.append(all_solutions[idx])
                sample_preds.append(all_preds[idx])
            
            ### Process predictions and scores using the 3D scoring function
            parsed_gt = parse_ground_truth(sample, args.data_name)
            new_gt, new_sample_preds, sample_scores = obtain_nHm_scores_and_preds(parsed_gt, sample_preds)
            sample.update({
                "think_solution": think_solutions[i * args.n_sampling : (i + 1) * args.n_sampling],
                "solution": sample_solutions,  # [n, 1, m]
                "gt": new_gt,
                "pred": new_sample_preds, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "score": sample_scores, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
            })

    # 3D CoT
    else:
        thinking_toks = [tokenizer.encode(think_solution.split("</think>")[0])[1:] for think_solution in think_solutions]
        
        ## Create prompts for different chunks of thinking
        prompts = [sample["prompt"] for sample in samples]
        prompt_chunk_thinks = []
        for r, thinking_tok in enumerate(thinking_toks):
            ### Cut thinking at different points, i.e. H 
            splits = [thinking_tok[: i * len(thinking_tok) // args.num_think_chunks] for i in range(1, args.num_think_chunks)]
            
            ### For each split, create multiple prompts for generating different solutions
            for split in splits:
                prompt_chunk_think = prompts[r//args.n_sampling] + tokenizer.decode(split) + "</think>\n\n"
                prompt_chunk_thinks.append(prompt_chunk_think)
        
        ## Generate solutions for all (except for the last) thinking chunks
        solution_sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            max_tokens=args.max_tokens_per_solution,
            n=args.num_solutions_per_chunk,
            skip_special_tokens=False,
            seed=args.seed,
        )
        llm_outputs = llm.generate(prompt_chunk_thinks, solution_sampling_params)
        llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
        nHm_solutions = [output.text for llm_output in llm_outputs for output in llm_output.outputs] # flatten
        assert len(nHm_solutions) == len(prompt_chunk_thinks) * args.num_solutions_per_chunk

        ## Also generate multiple solutions for the full thinking
        prompt_full_thinks = []
        for r, think_solution in enumerate(think_solutions):
            if "</think>" not in think_solution:
                prompt_full_think = prompts[r//args.n_sampling] + think_solution + "\n</think>\n\n"
            else:
                prompt_full_think = prompts[r//args.n_sampling] + think_solution.split("</think>")[0] + "</think>"
            prompt_full_thinks.append(prompt_full_think)

        ## Generate additional solutions for the full thinking
        if args.num_solutions_per_chunk > 1:
            solution_sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                min_p=args.min_p,
                max_tokens=args.max_tokens_per_solution,
                n=args.num_solutions_per_chunk - 1, # minus 1 because we already have one
                skip_special_tokens=False,
                seed=args.seed,
            )
            llm_outputs = llm.generate(prompt_full_thinks, solution_sampling_params)
            llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
            nm_solutions = [output.text for llm_output in llm_outputs for output in llm_output.outputs] # flatten
            assert len(nm_solutions) == len(prompt_full_thinks) * (args.num_solutions_per_chunk - 1)
        
        ## Organize all the solutions into a 3D structure: [n, H, m]
        all_solutions = []
        all_preds = []
        
        chunk_idx = 0
        final_idx = 0
        for s in range(len(think_solutions)):  # For each initial sample
            sample_solutions = []
            sample_preds = []
            
            ### Get completions for intermediate chunks
            for c in range(args.num_think_chunks - 1):  # For each chunk except the last
                think_chunk_solutions = []
                think_chunk_preds = []
                
                for a in range(args.num_solutions_per_chunk):  # For each solution
                    solution = nHm_solutions[chunk_idx]                
                    pred = extract_pred_and_parse(solution, args.data_name)
                    think_chunk_solutions.append(solution)
                    think_chunk_preds.append(pred)
                    chunk_idx += 1
                
                sample_solutions.append(think_chunk_solutions)
                sample_preds.append(think_chunk_preds)
            
            ### Get solutions for final chunk (full thinking)
            final_solutions = [n_solutions[s]]  # Start with the original solution
            final_preds = [extract_pred_and_parse(n_solutions[s], args.data_name)]
            if args.num_solutions_per_chunk > 1:
                for a in range(args.num_solutions_per_chunk - 1):
                    solution = nm_solutions[final_idx]
                    pred = extract_pred_and_parse(solution, args.data_name)
                    final_solutions.append(solution)
                    final_preds.append(pred)
                    final_idx += 1
            
            sample_solutions.append(final_solutions)
            sample_preds.append(final_preds)
            
            all_solutions.append(sample_solutions)
            all_preds.append(sample_preds)
        
        ## Process results and put back into samples
        for i, sample in enumerate(samples):
            sample_solutions = []
            sample_preds = []
            
            for s in range(args.n_sampling):
                idx = i * args.n_sampling + s
                sample_solutions.append(all_solutions[idx])
                sample_preds.append(all_preds[idx])

            ### Process predictions and scores using the 3D scoring function
            parsed_gt = parse_ground_truth(sample, args.data_name)
            new_gt, new_sample_preds, sample_scores = obtain_nHm_scores_and_preds(parsed_gt, sample_preds)
            sample.update({
                "think_solution": think_solutions[i * args.n_sampling : (i + 1) * args.n_sampling],
                "solution": sample_solutions,  # [n, H, m]
                "gt": new_gt,
                "pred": new_sample_preds, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "score": sample_scores, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
            })

    # save outputs
    print(f"Saved 3D CoT to {out_file}")
    with open(out_file, "w") as f:
        json.dump(samples, f, indent=2)
    

def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", default="./benchmarks", type=str)
    parser.add_argument("--data_name", default="aime24", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    
    # Model and sampling
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=40000)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--min_p", default=0., type=float)
    parser.add_argument("--temperature", default=0, type=float)

    # 3D CoT
    parser.add_argument("--n_sampling", default=1, type=int, help="I.e. n")
    parser.add_argument("--num_think_chunks", default=1, type=int,
                        help="Evenly split the original think into multiple chunks, i.e. H")
    parser.add_argument("--num_solutions_per_chunk", default=1, type=int,
                       help="Number of solution variations to generate for each reasoning chunk, i.e. m")
    parser.add_argument("--max_tokens_per_solution", default=2048, type=int)
    
    # Think and solutions saving/loading
    parser.add_argument("--save_think_solutions", action="store_true", default=False,
                       help="Save think and solutions to a file for later reuse")
    parser.add_argument("--load_think_solutions", type=str, default="",
                       help="Path to a file containing think solutions to load instead of generating")

    # Save
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    main(args)