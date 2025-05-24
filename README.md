<div align="center">

# ⛓️ Fractured Chain-of-Thought Reasoning ⛓️


[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.12992) 
[![Github](https://img.shields.io/badge/Fractured%20CoT-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/BaohaoLiao/frac-cot)
[![Website](https://img.shields.io/badge/Blog-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://huggingface.co/spaces/Salesforce/Efficient-Reasoning)

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📢 Introduction</a> •
    <a href="#support" style="text-decoration: none; font-weight: bold;">🤝 Support</a>
  </p>
  <p>
    <a href="#installation" style="text-decoration: none; font-weight: bold;">⚙️ Installation</a> •
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">🔎 Evaluation</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">📝 Citation</a>
  </p>
</div>

</div>


# 🎉News
- **[2025.05.21]** We release the first version that mainly supports the pass@k experiments.

# 📢Introduction
We introduce **Fractured Sampling**, a unified inference-time strategy that interpolates between full CoT and solution-only sampling along three orthogonal axes: (1) the number of reasoning trajectories (n dimension), (2) the number of final solutions per trajectory (m dimension), and (3) the depth at which reasoning traces are truncated (H dimension). 
<p align="center">
  <img src="figs/frac_cot.gif" width="70%" />
</p>

**Main Takeaways**
1. 📈 The long-reasoning LLM is able to use truncated CoT to derive a correct solution.
2. 🌟 We can sample in 3D: the full-CoT (n dimensin), the solution (m dimension) and the truncated CoT (H dimension). The H dimension shows the steepest log-linear scaling gains in Pass@k.
3. 🚀 Sampling over all 3 dimensions offers the highest Pass@k.
4. 📊 We can use a process reward model to select the best solution among all 3D samplings, a ~10% accuracy improvement compared to only sampling in the n dimension.
5. 🧠 We can use the self-consistency property within the H dimension to early stop the genration, saving 20% tokens without sacrifying accuracy.


# 🤝Support
- [x] Pass@k for single dimension and multiple dimensions.
- [ ] Best-of-N accross multiple dimensions.
- [ ] Early stopping for efficient generation.

# ⚙️Installation
```bash
conda create -n frac_cot python=3.10
conda activate frac_cot
pip install -r requirements.txt
```

# 🔎Evaluation
1. Pass@k for single and multiple dimensions.
    ```bash
    # Generation
    bash ./scripts/gen.sh
    # Evaluate
    bash ./scriots/passk.sh
    ```

# 📝Citation
If you find our work useful, please cite as:
```
@article{liao2025fractured,
  title={Fractured Chain-of-Thought Reasoning},
  author={Liao, Baohao and Dong, Hanze and Xu, Yuhui and Sahoo, Doyen and Monz, Christof and Li, Junnan and Xiong, Caiming},
  journal={arXiv preprint arXiv:2505.12992},
  year={2025}
}
```