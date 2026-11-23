"""
LLM-based representation of the benchmark molecules: a 768-dimensional embedding of each
molecule's canonical SMILES from pretrained chemical language model MoLFormer-XL

    smiles.csv -> llm_based.csv         i, emb_000 ... emb_767 (float)

STEPS
  1. Canonical SMILES read from simles.csv (written by build_string_based.py); same RDKit
     canonicalization hydrogen implicit, aromaicity percieved).
  2. SMILES tokenized with the MoLFormer tokenizer (atom-level SIMILES tokens, max 200 tokens (enough
    for small molecules).
  3. Tokens passed through MoLFormer-XL (ibm/MoLFormer-XL-both-10pct; Hugging Face):
    transformer encoder with linear attention and toray position ebeddings, pretraiend by masked-token
    prediction on ~110M SMILES.
    Used frozen, inference only: no training of fine-tuning.
    deterministic_eval=True so the embedding is reproducible run to run.
  4. The pooled output of the encoder gives one 768-feature vector per molecule -> llm_based.csv

NOTES
  * The embedding is continuous and learned: individual dimensions have no chemical meaning.
  * The vector dependson the SMILES string, so the canonical SMILES isused to give one vector per 
  molecule.
  * H2 is written with explicit hydrogens ([H][H]), a form rarely seen in pretraining; its embedding may
  be less reliable than the others.
  * The model code is lodaded with trust_remote_code=True; newer transformers versions may break it (works with transoremers <4.40)

Molecules = every id  in similes.csv
Run:  python build_llm_based.py     (needs torch, transformers, rdkit)
"""


from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


HERE = Path(__file__).resolve().parent
IN_CSV = HERE / "smiles.csv"
OUT_CSV = HERE / "llm_based.csv"

MODEL_NAME = "IBM/MoLFormer-XL-both-10pct"
MAX_TOKENS = 202        #MoLFormer positional limit
BATCH_SIZE = 32



def load_smiles(path):
    """
    id and canonical SMILES from smiles.csv (columns: id, smiles).
    """

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id", df.columns[0])
    smi_col = cols.get("smiles", df.columns[1])

    return df[id_col].astype(str).tolist(), df[smi_col].astype(str).tolist()


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True).eval()

    return tok, model


def check_lengths(tok, ids, smiles):
    """ 
    Stop if any SMILES would be truncated.
    """
    too_long = [(i, len(tok(s)["input_ids"])) for i, s in zip(ids, smiles) 
                if len(tok(s)["input_ids"]) > MAX_TOKENS]

    if too_long:
        raise ValueError(f"SMILES longer than {MAX_TOKENS} tokens: {too_long}")


def embed(tok, model, smiles):
    """
    One pooled 768-d vector per SMILES.
    """
    vecs = []
    for i in range(0, len(smiles), BATCH_SIZE):
        batch = tok(smiles[i:i + BATCH_SIZE], padding=True, return_tensors="pt")
        with torch.no_grad():
            vecs.append(model(**batch).pooler_output)

    return torch.cat(vecs).numpy()


def main():
    ids, smiles = load_smiles(IN_CSV)
    tok, model = load_model()
    check_lengths(tok, ids, smiles)

    X = embed(tok, model, smiles)
    cols = [f"emb_{k:03d}" for k in range(X.shape[1])]
    out = pd.DataFrame(X, columns=cols)
    out.insert(0, "id", ids)
    out.to_csv(OUT_CSV, index=False, float_format="%.6f")

    print(f"{len(ids)} molecules, {X.shape[1]} dimensions -> {OUT_CSV.name}")


if __name__ == "__main__":
    main()
