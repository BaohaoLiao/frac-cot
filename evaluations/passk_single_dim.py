import json
import argparse
import numpy as np
from utils.eval import pass_at_k


def calculate_pass_at_k_n_dimension(all_sub_scores, k_values):
    """
    Calculate Pass@k across the n dimension
    Uses the final chunk and first answer: sub_scores[:, -1, 0]
    Input:
        all_sub_scores: #questions x n x H x m
    """
    all_n_scores = []
    for sub_scores in all_sub_scores:
        all_n_scores.append([scores[-1][0] for scores in sub_scores])

    all_n_correct = [sum(n_scores) for n_scores in all_n_scores]
    n_total = len(all_n_scores[0])

    # Calculate Pass@k for each k value
    results = {}
    for k in k_values:
        pass_k = np.mean([pass_at_k(n_total, n_correct, k) for n_correct in all_n_correct])
        results[f"pass@{k}"] = float(f"{pass_k:.4f}")

    return results


def calculate_pass_at_k_H_dimension(all_sub_scores, k_values):
    """
    Calculate Pass@k across the H dimension
    Evenly samples k-1 chunks from each sample's reasoning and always includes the last chunk, 
    e.g. sample from sub_scores[0, :, 0]
    Averages the Pass@k scores across all samples, i.e. over dimension n
    Input:
        all_sub_scores: #questions x n x H x m
    """
    n_samples = len(all_sub_scores[0])
    H_chunks = len(all_sub_scores[0][0])

    # For each k value
    results = {}
    for k in k_values:
        sample_H_indices = np.linspace(H_chunks-1, H_chunks//k-1, k, dtype=int)[::-1]
        pass_k = []

        # for each question
        for sub_scores in all_sub_scores:
            pass_k_for_each_n = []

            # for each n dimension
            for n in range(n_samples):
                H_scores = [sub_scores[n][h][0] for h in sample_H_indices]
                H_correct = sum(H_scores)
                pass_k_for_each_n.append(pass_at_k(k, H_correct, k))

            pass_k.append(np.mean(pass_k_for_each_n))

        results[f"pass@{k}"] = float(f"{np.mean(pass_k):.4f}")
    
    return results


def calculate_pass_at_k_m_dimension(all_sub_scores, k_values):
    """
    Calculate Pass@k across the m dimension
    Uses the final chunk and all answers: sub_scores[:, -1, :]
    Averages the Pass@k scores across all samples
    Input:
        all_sub_scores: #questions x n x H x m
    """
    m_total = len(all_sub_scores[0][0][0])
    n_samples = len(all_sub_scores[0])

    # Calculate Pass@k for each k value
    results = {}
    for k in k_values:
        pass_k = []

        # for each question
        for sub_scores in all_sub_scores:
            pass_k_for_each_n = []

            # for each n dimension
            for n in range(n_samples):
                m_correct = sum(sub_scores[n][-1])
                pass_k_for_each_n.append(pass_at_k(m_total, m_correct, k))

            pass_k.append(np.mean(pass_k_for_each_n))

        results[f"pass@{k}"] = float(f"{np.mean(pass_k):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate Pass@k metrics for mathematical reasoning evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the evaluation output JSON file")
    parser.add_argument("--dimension", type=str, choices=['n', 'H', 'm'], required=True, 
                        help="Dimension to calculate Pass@k for (n: sampling, H: think chunks, m: solutions per chunk)")
    parser.add_argument("--k_values", type=str, default="1,2,4,8,16", 
                        help="Comma-separated list of k values to calculate Pass@k for")
    args = parser.parse_args()
    
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Load samples
    with open(args.input_file, 'r') as f:
        samples = json.load(f)

    # Extract sub_scores from all samples
    all_scores = [sample['score'] for sample in samples]
    n, H, m = len(all_scores[0]), len(all_scores[0][0]), len(all_scores[0][0][0]) 

    print(f"  #Questions: {len(all_scores)}")
    print(f"  n: {n}")
    print(f"  H: {H}")
    print(f"  m: {m}")
    print("=" * 50)

    # Calculate Pass@k based on the specified dimension
    if args.dimension == 'n':
        assert max(k_values) <= n, "k should not be larger than n"
        results = calculate_pass_at_k_n_dimension(all_scores, k_values)
    elif args.dimension == 'H':
        assert max(k_values) <= H, "k should not be larger than H"
        results = calculate_pass_at_k_H_dimension(all_scores, k_values)
    else:
        assert max(k_values) <= m, "k should not be larger than m"
        results = calculate_pass_at_k_m_dimension(all_scores, k_values)

    print(results)


if __name__ == "__main__":
    main()