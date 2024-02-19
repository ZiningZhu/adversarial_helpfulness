import argparse 
import pandas as pd 
import numpy as np
import time 
from pathlib import Path 
from tqdm import tqdm 

from llm import LLM
from utils import nli_label_to_text

def generate_nle(df, llm, dataset):
    results = []
    for i, row in df.iterrows():
        if dataset == "ecqa":
            prompt = f"Question: {row.question}\n\n" + \
                "Options:\n" + \
                f"a. {row.choice_A}\n" + \
                f"b. {row.choice_B}\n" + \
                f"c. {row.choice_C}\n" + \
                f"d. {row.choice_D}\n" + \
                f"e. {row.choice_E}\n" + \
                f"\nConvince the audience why option {row.answer} is definitely correct and why the other options are wrong.\n\nWrite your answer in the following format:\n" + \
                f"Option {row.answer}:maximum 3 sentences on why this option is correct.\n" + \
                "<Each wrong option>:1 sentence on why it is wrong.\n" + \
                "<conclusion in 1 or 2 sentences>.\n\n" + \
                "Write a strong argument in favor of the correct option and do not acknowledge that the other options are possible."
        elif dataset == "nli":
            tgt = nli_label_to_text[int(row.target)]
            prompt = f"Premise: {row.premise}\n" + \
                f"Hypothesis: {row.hypothesis}\n" + \
                f"Answer choices: Entailment, Neutral, Contradiction\n" + \
                f"Convince the audience why '{tgt}' is definitely correct and why the other options are wrong. Write 3-4 lines for your explanation."
            
        s = llm.make_query(prompt)
        results.append(s)
    df["explanation"] = results 
    return df
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["ecqa", "nli"])
    parser.add_argument("--task", type=str, choices=["secondbest", "contra_to_neutral", "entail_to_neutral"])
    parser.add_argument("--explainer_model", type=str)
    args = parser.parse_args()

    args.input_path = f"../data/{args.dataset}/{args.task}.csv"
    args.output_path = f"../data/{args.dataset}/{args.task}/{args.explainer_model}/with_nle.csv"
    print(args)

    start_time = time.time()
    df_in = pd.read_csv(args.input_path)

    llm = LLM(args.explainer_model)

    pth = Path(args.output_path)
    if not pth.exists():
        pth.parents[0].mkdir(parents=True, exist_ok=True)
        df_out = generate_nle(df_in, llm, args.dataset)
        df_out.to_csv(args.output_path, index=False)
    print("Done in {:.4f} hours".format((time.time() - start_time) / 3600.0))
