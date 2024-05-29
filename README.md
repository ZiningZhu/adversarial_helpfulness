# LLM-generated Black-box Explanations can be Adversarially Helpful

Arxiv: [![](https://img.shields.io/badge/arXiv-2405.06800-b31b1b.svg)](https://arxiv.org/abs/2405.06800)

The data are at [![](https://img.shields.io/badge/S3-blue.svg)](https://warm-snowball-public-datasets.s3.amazonaws.com/adversarial_helpfulness/data.zip)

## Scripts for reproducing the findings
- Section 4 **Human evaluation**: `notebooks/analyze_human_and_proxy_results.ipynb` (this notebook also collects the automatic evaluation results)  
- Section 5 **Automatic evaluation**: `scripts/proxy_evaluation.py`  
- Section 6 **Strategies** `scripts/persuasion_strategy.py`  
- Section 7 **A structural analysis** `notebooks/generate_graph_example.ipynb`  

## Other scripts
- `notebooks/check_proxy_completeness.ipynb` is used to check the completeness of experiment results.
- `scripts/llm.py` contains a tool for prompting LLMs.

## Citation

```
@article{ajwani2024generated,
    title={{LLM-generated Black-box Explanations can be Adversarially Helpful}},
    author={Ajwani, Rohan and Javaji, Shashidhar Reddy and Rudzicz, Frank and Zhu, Zining},
    journal={arXiv preprint arXiv:2405.06800},
    year={2024}
}
```