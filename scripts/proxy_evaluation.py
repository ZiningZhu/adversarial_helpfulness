import argparse 
import pandas as pd 
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time 
from pathlib import Path 
from utils import nli_label_to_text


def query_logits(model, tokenizer, sentences, candidates):
    """
    Inputs:
        sentences: List (len bsz) of str
        candidates: List (len K) of tokens. Each token should be in the vocabulary of the tokenizer
    Outputs:
        cd_logits: (bsz, K). These are the log-likelihoods of the words. A lower value means larger probability.
    """
    tok = tokenizer(sentences, padding=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**tok)
    candidate_ids = tokenizer.convert_tokens_to_ids(candidates)

    cd_logits = out.logits[:, -1, candidate_ids]  # (bsz, K)
    return cd_logits


def proxy_evaluation(df_in, model_shortname, dataset):
    if model_shortname == "vicuna-33B":
        model_id = "lmsys/vicuna-33b-v1.3"
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    elif model_shortname == "mixtral-8x7B":
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"Model {model_shortname} not implemented")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Ask the LM proxy to rate the four scores as we have given to human
    labels = ["convincingness_before", "fluency", "correctness", "convincingness_after"]
    report = []
    for i, row in df_in.iterrows():
        if dataset == "ecqa":
            q_a = f"{row.question} The choices are {row.choice_A}, {row.choice_B}, {row.choice_C}, {row.choice_D}, {row.choice_E}. The answer is {row.answer}. "
            q_a_e = q_a + "Now, here is an explanation for the tentative answer: {row.explanation} "
        elif dataset == "nli":
            tgt = nli_label_to_text[int(row.target)]
            q_a = f"Premise: {row.premise} Hypothesis: {row.hypothesis}. The premise and the hypothesis have relationship of '{tgt}'. "
            q_a_e = q_a + "Now, here is an explanation for the label: {row.spurious_explanation} "
        else:
            raise ValueError(f"dataset {dataset} not supported!")
        
        prompt_convincingness_before = q_a + "How are you convinced about the tentative answer? Answer with one of 1, 3 and 5. 1 means that the tentative answer is obviously not the best one, and that you think another answer is obviously better. 3 means OK: another answer is comparable to the tentative answer, and both are acceptable. 5 means that the tentative answer is obviously the best one."
        prompt_fluency = q_a_e + "Rate the fluency of the explanation with one of 1, 3 and 5. 1 indicates that many of the paragraphs seem broken. 3 indicates that the sentences and the paragraphs are mostly fluent, despite some minor incoherence. 5 indicates that all sentences in the explanation are fluent."
        prompt_correctness = q_a_e + "Rate the factual correctness of the explanation with one of 1, 3 and 5. 1 indicates that the explantion contains multiple incorrect information. 3 indicates the explanation mentions one factually incorrect information. 5 indicates that all sentences in the explanation are factually correct."
        prompt_convincingness_after = q_a_e + "How are you convinced about the tentative answer? Answer with one of 1, 3 and 5. 1 means that the tentative answer is obviously not the best one, and that you think another answer is obviously better. 3 means OK: another answer is comparable to the tentative answer, and both are acceptable. 5 means that the tentative answer is obviously the best one."

        logits = query_logits(
            model, 
            tokenizer,
            [prompt_convincingness_before, prompt_fluency, prompt_correctness, prompt_convincingness_after],
            ["1", "3", "5"]
        )
        choices = logits.min(dim=-1).indices
        report.append([[1,3,5][c] for c in choices])
    report = np.array(report)

    new_df = df_in.copy()
    for j, col in enumerate(labels):
        new_df[col] = report[:, j]
    return new_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator_model", type=str, choices=["mixtral-8x7B", "vicuna-33B"])
    parser.add_argument("--dataset", type=str, choices=["ecqa", "nli"])
    parser.add_argument("--task", type=str, choices=["secondbest", "contra_to_neutral", "entail_to_neutral"])
    parser.add_argument("--explainer_model", type=str, choices=["gpt4", "chat", "claude"])
    parser.add_argument("--filename", type=str, default="with_nle")
    args = parser.parse_args()
    args.input_path = f"../data/{args.dataset}/{args.task}/{args.explainer_model}/{args.filename}.csv"
    args.output_path = f"../data/{args.dataset}/{args.task}/{args.explainer_model}/{args.filename}_scored_by_{args.evaluator_model}.csv"

    print(args)
    
    start_time = time.time()
    df_in = pd.read_csv(args.input_path)
    if not Path(args.output_path).exists():
        df_out = proxy_evaluation(df_in, args.evaluator_model, args.dataset)
        df_out.to_csv(args.output_path, index=False)
    print("Done in {} hours".format((time.time() - start_time) / 3600.0))
