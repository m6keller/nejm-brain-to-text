import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import editdistance  # pip install editdistance
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate_model_helpers import remove_punctuation
# ==================================================

# ================= CONFIGURATION =================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_PATH = "validation_ground_truth.csv"
BATCH_SIZE = 8
# =================================================

def calculate_wer_ref_style(references, hypotheses):
    """
    Calculates WER using the imported remove_punctuation logic + editdistance.
    
    We explicitly .lower() here to ensure we match the ~19% WER baseline 
    found in your evaluation script.
    """
    total_edit_distance = 0
    total_true_length = 0

    for ref, hyp in zip(references, hypotheses):
        ref_clean = remove_punctuation(ref).strip().lower()
        hyp_clean = remove_punctuation(hyp).strip().lower()

        ref_words = ref_clean.split()
        hyp_words = hyp_clean.split()

        ed = editdistance.eval(ref_words, hyp_words)

        total_edit_distance += ed
        total_true_length += len(ref_words)

    if total_true_length == 0:
        return 0.0

    return 100 * total_edit_distance / total_true_length

def load_text(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"{path} must contain 'id' and 'text' columns.")
    return df.set_index("id")["text"].astype(str)

def load_llm(model_name, device):
    """
    Load the language model & tokenizer
    """
    print(f"Loading LM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", 
        device_map=device              
    )
    model.eval()
    return model, tokenizer

def lm_score_sentences(sent_list, model, tokenizer, batch_size=8):
    """
    Compute a score for each sentence in sent_list using the LM.
    """
    scores = []
    total = len(sent_list)

    for i in range(0, total, batch_size):
        batch = sent_list[i:i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            log_probs = F.log_softmax(out.logits.float(), dim=-1)

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        for b in range(len(batch)):
            length = int(attn[b].sum().item())
            total_lp = 0.0
            for t in range(1, length):
                total_lp += log_probs[b, t - 1, input_ids[b, t]].item()
            scores.append(total_lp)

    return scores

def best_sentence(model_text_lists, model_score_lists):
    '''
    Select the best sentence per item across multiple models
    '''
    num_models = len(model_text_lists)
    if num_models == 0:
        raise ValueError("No models provided for ensembling.")

    num_items = len(model_text_lists[0])

    chosen = []
    chosen_from = []

    for idx in range(num_items):
        scores = [model_score_lists[m][idx] for m in range(num_models)]
        best_model_idx = max(range(num_models), key=lambda m: scores[m])

        chosen.append(model_text_lists[best_model_idx][idx])
        chosen_from.append(best_model_idx + 1) # 1-based index for logging

    return chosen, chosen_from

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble multiple prediction CSVs using LLM scoring.")
    
    parser.add_argument("--candidates", nargs='+', required=True, 
                        help="List of model output CSVs to ensemble (space separated).")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save the ensembled output CSV.")
    parser.add_argument("--target", type=str, default=TARGET_PATH,
                        help="Path to the ground truth CSV (must contain 'id' and 'text').")
    parser.add_argument("--model", type=str, default=MODEL_NAME, 
                        help="HuggingFace model name or path.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, 
                        help="Batch size for LLM inference.")

    args = parser.parse_args()

    # 1. Load Target
    print(f"Loading target from {args.target}")
    target = load_text(args.target)

    # 2. Load Candidate Models
    models = []
    for path in args.candidates:
        print(f"Loading model output from {path}")
        m = load_text(path)
        models.append(m)

    # 3. Align all models to the target index
    models = [m.reindex(target.index) for m in models]
    
    target_ids = target.index.tolist()
    target_texts = target.tolist()
    model_text_lists = [m.tolist() for m in models]

    # 4. Load LLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model, llm_tokenizer = load_llm(args.model, device=device)

    # 5. Score model outputs
    model_score_lists = []
    for i, texts in enumerate(model_text_lists, start=1):
        texts = [str(t) if pd.notna(t) else "" for t in texts]
        
        print(f"Scoring model {i} ({len(texts)} sentences)...")
        scores = lm_score_sentences(texts, llm_model, llm_tokenizer, batch_size=args.batch_size)
        model_score_lists.append(scores)

    # 6. Pick best sentence per id
    print("Selecting best sentence per id...")
    chosen_texts, chosen_sources = best_sentence(
        model_text_lists,
        model_score_lists
    )

    # 7. Save Output
    df_out = pd.DataFrame({
        "id": target_ids,
        "text": chosen_texts,
        "chosen_model": chosen_sources
    })
    df_out.to_csv(args.output, index=False)
    print(f"Ensemble text output written to {args.output}")

    # 8. Compute WER (Using imported remove_punctuation + local loop logic)
    print("Computing WER before and after LLM ensembling...")
    
    target_list = [str(t) for t in target_texts]

    for i, candidate_texts in enumerate(model_text_lists, start=1):
        candidate_list = [str(t) for t in candidate_texts]
        wer = calculate_wer_ref_style(target_list, candidate_list)
        print(f"WER candidate {i} ({args.candidates[i-1]}): {wer:.2f}%")

    ensemble_wer = calculate_wer_ref_style(target_list, chosen_texts)
    print(f"WER after LLM ensembling: {ensemble_wer:.2f}%")