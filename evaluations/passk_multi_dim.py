import json
import argparse
import numpy as np
from utils.eval import pass_at_k


def main():
    parser = argparse.ArgumentParser(description='Calculate pass@k in the dimension of n.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the evaluation output JSON file')
    parser.add_argument("--k_values", type=str, default="1,2,4,8", help='k value for maj@k and pass@k calculation')
    parser.add_argument('--h_chunks', type=int, default=1, 
                        help='Number of chunks to use from H dimension. If -1, use all predictions at H dimension.')
    parser.add_argument('--m_solutions', type=int, default=1, help='Number of solutions to use from m dimension')
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

    # Extract preds and gt from all samples
    all_preds = [sample['pred'] for sample in samples]
    all_gts = [sample['gt'] for sample in samples]
    
    num_qs, n, H, m = len(all_preds), len(all_preds[0]), len(all_preds[0][0]), len(all_preds[0][0][0])
    assert max(k_values) <= n, "k should not be larger than n"
    assert args.h_chunks <= H, "h_chunks should not be larger than H"
    assert args.m_solutions <= m, "m_solutions should not be larger than m"

    print(f"  #Questions: {num_qs}")
    print(f"  n: {n}")
    print(f"  H: {H}")
    print(f"  m: {m}")
    print("=" * 50)
    
    pass_k_results = {}
    for k in k_values:
        all_pass_ks = []

        for q_idx in range(num_qs):
            q_sub_preds = all_preds[q_idx]  # n x H x m
            q_gt = all_gts[q_idx]

            # Reshape to n x h_chunks*m_solutions
            q_preds = []
            H_indices = np.linspace(H-1, H//args.h_chunks-1, args.h_chunks, dtype=int)[::-1] # must include the last pred
            m_indices = np.arange(args.m_solutions)
            for n_idx in range(n):
                tmp = []
                for h_idx in H_indices:
                    for m_idx in m_indices:
                        tmp.append(q_sub_preds[n_idx][h_idx][m_idx])
                q_preds.append(tmp)

            # Calculate pass@k
            q_scores = [sum([q_pred == q_gt for q_pred in q_preds[i]])>=1 for i in range(n)]
            all_pass_ks.append(pass_at_k(n, sum(q_scores), k))

        pass_k_results[f"pass@{k}"] = float(f"{np.mean(all_pass_ks):.4f}")

    print(pass_k_results)


if __name__ == "__main__":
    main()