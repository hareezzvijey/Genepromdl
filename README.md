# GenePromDL

**Drug Sensitivity Prediction Using Promoter DNA, Methylation, and Tissue Context**

A systematic deep learning investigation into which biological features carry predictive signal for cancer drug sensitivity, built on GDSC and CCLE data.

---

## What This Project Does

GenePromDL trains a four-branch neural network to predict how sensitive a cancer cell line will be to a drug, given:

- The drug's **chemical structure** (Morgan fingerprint)
- The gene's **promoter DNA sequence** (1200bp around TSS)
- The gene's **methylation state** (CCLE RRBS beta-value)
- The cell line's **tissue / cancer type** (from GDSC cell line metadata)

The project ran three systematic experiments and documented two negative results and one positive result — forming a research narrative about what actually drives drug sensitivity prediction at the promoter level.

---

## Key Results

| Version | Task | Label | Best Metric | Drug Contributes? |
|---------|------|-------|-------------|-------------------|
| V1 | Binary classification | Methylation state (β > 0.6) | AUROC = 0.96 | ❌ No (Δ = 0.0016) |
| V2 | Regression | Z_SCORE, no tissue | R² = 0.000 | ❌ No signal |
| V3 | Regression | Z_SCORE + tissue | R² = 0.226 | ✅ Yes (18.3%) |

### V3 Final Metrics

| Split | RMSE | MAE | R² | Pearson r |
|-------|------|-----|----|-----------|
| Train | 0.861 | 0.667 | 0.282 | 0.532 |
| Val   | 0.901 | 0.701 | 0.214 | 0.466 |
| Test  | 0.891 | 0.687 | 0.226 | 0.476 |

### Input Contribution Test (V3)

| Input Branch | Mean Prediction Change (shuffle) | Relative Contribution |
|---|---|---|
| Drug fingerprint | 0.1196 | 18.3% |
| DNA sequence | 0.0000 | 0.0% |
| Methylation beta | 0.0114 | 1.7% |
| **Tissue / Cancer type** | **0.5236** | **80.0%** |

### Ablation Study

| Input Combination | Test R² |
|---|---|
| Drug only | 0.0002 |
| Sequence only | 0.0000 |
| Methylation only | 0.0017 |
| Tissue only | 0.2196 |
| Drug + Tissue | ~0.215 |
| **Full model (all 4)** | **0.2255** |

---

## Core Scientific Findings

**1. High accuracy can mask the complete absence of drug signal.**
V1 achieved AUROC=0.96, but the drug shuffle test showed the model was ignoring drug input entirely (Δ=0.0016). The label was a function of gene+cell identity, not drug treatment — the model correctly learned this and discarded the drug branch. Standard accuracy metrics would not have caught this.

**2. Tissue/cancer-type context is the dominant driver of drug sensitivity.**
Across all experiments, tissue type alone explains ~22% of drug sensitivity variance — nearly as much as the full four-input model. This aligns with pharmacogenomics literature: cancer lineage is the strongest determinant of drug response when gene expression and mutation data are absent.

**3. Drug chemical structure provides modest but real signal (+18.3%).**
When tissue context is included, the Morgan fingerprint branch contributes meaningfully. Drug structure matters — but only when the cellular context that determines the response is also available.

**4. Promoter DNA sequence does not contribute to sensitivity prediction.**
Despite dilated CNNs, self-attention, and SENET over 1200bp sequences, the sequence branch contributes 0.0% in V3. Promoter sequence is identical across cell lines for the same gene — it cannot encode cell-line-specific drug response. Gene expression or mutation data would be needed to make sequence-level features useful here.

---

## Model Architecture (V3)

```
Drug fingerprint (2048,)       Promoter sequence (4, 1200)
        │                               │
  Dense(512) → BN → ReLU         Embed Conv1d(16)
  Dense(256) → BN → ReLU         DilatedConv1d(64,  d=1)
        │                         DilatedConv1d(128, d=2)
        │                         DilatedConv1d(256, d=4)
        │                         SelfAttention (pool→50)
        │                         SENET
        │                         GlobalMaxPool
        │                               │
Meth beta (1,)              Tissue one-hot (tissue_dim,)
        │                               │
  Linear(1→32)→16             Linear(→64)→BN→Linear(→32)
        │                               │
        └──────────── Concat ───────────┘
                          │
                    Dense(256) → BN → Dropout
                    Dense(64)  → BN → Dropout
                    Linear(64 → 1)
                          │
                      Z_SCORE (regression)
```

**Architecture notes:**
- Self-attention pools sequence from 1200→50 positions before computing attention — reduces VRAM from ~13GB to ~0.1GB per batch (essential for 8GB VRAM GPUs)
- Odd kernel sizes (7, 3) throughout to avoid PyTorch padding warnings
- Loss: MSELoss | Optimizer: Adam(1e-4) | EarlyStopping patience=5

---

## Dataset

Six public databases, no proprietary access required:

| Source | Data | Used For |
|---|---|---|
| [GDSC](https://www.cancerrxgene.org) | Drug sensitivity (Z_SCORE) | Regression target |
| [CCLE / DepMap](https://depmap.org) | RRBS methylation beta-values | Input covariate |
| [DepMap Model.csv](https://depmap.org) | COSMIC→CCLE cell line bridge | ID mapping |
| [PubChem API](https://pubchem.ncbi.nlm.nih.gov) | Drug SMILES strings | Morgan fingerprints |
| [Ensembl REST API](https://rest.ensembl.org) | Promoter DNA sequences | Sequence input |
| GDSC Cell Line Details | Tissue type, cancer type, MSI | Tissue features |

**V3 training dataset:**
- 1,119,100 samples · 229 drugs · 100 genes · 556 cell lines
- Split 70/15/15 **by drug name** (test set contains completely unseen drugs)
- Stored as memory-mapped `.dat` files (~7.7GB total) to avoid RAM overflow

---

## Project Structure

```
genepromdl/
├── src/
│   ├── config.py                  # All paths and hyperparameters
│   ├── 00_rebuild_v3.py           # Build V3 dataset with tissue features
│   ├── 01_split_by_drug_v3.py     # Drug-disjoint train/val/test split
│   ├── dataset_module_v3.py       # PyTorch Dataset + DataLoader
│   ├── 02_model_v3.py             # Architecture test
│   ├── model_module_v3.py         # Model definition
│   ├── 03_train_v3.py             # Training loop (GPU)
│   ├── 04_evaluate_v3.py          # Metrics + 4-way contribution test
│   └── 05_ablation_v3.py          # All input combination ablation
├── results_v3/                    # Auto-created: checkpoints, plots, CSVs
├── requirements.txt
└── README.md
```

---

## Setup and Running

### 1. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn scipy matplotlib tqdm psutil openpyxl
```

### 2. Verify GPU

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 3050
```

### 3. Place data files

```
data/
├── GDSC_DATASET.csv
├── GDSC2-dataset.csv
├── Compounds-annotation.csv
├── Cell_Lines_Details.xlsx
├── Model.csv
├── CCLE_RRBS_TSS_1kb_20180614.txt
├── smiles_cache.pkl           # pre-built from PubChem
├── drug_fps.pkl               # pre-built Morgan fingerprints
├── seq_cache.pkl              # pre-built Ensembl sequences
└── gene_ohe.pkl               # pre-built one-hot sequences
```

### 4. Run the pipeline

```bash
python src/00_rebuild_v3.py          # ~15 min — builds dataset
python src/01_split_by_drug_v3.py    # < 1 min — creates split indices
python src/02_model_v3.py            # < 1 min — tests architecture
python src/03_train_v3.py            # ~45 min — trains on GPU
python src/04_evaluate_v3.py         # ~10 min — full evaluation
python src/05_ablation_v3.py         # ~3 hrs  — all 15 combinations
```

### 5. If you get CUDA out of memory

Open `config.py` and reduce:

```python
BATCH_SIZE = 128   # default 256 — halve if OOM
```

---

## Hardware

| Component | Spec |
|---|---|
| CPU | Intel Core i7 |
| GPU | NVIDIA RTX 3050 (8GB VRAM) |
| Training time | ~250 sec/epoch at batch_size=256 |
| Total training | ~2,500 sec (10 epochs, early stopping) |
| Peak VRAM | ~6.2 GB |

---

## Limitations

- No gene expression or mutation data — adding these would substantially improve R² (literature shows R²=0.4–0.6 with these features)
- CCLE methylation is a baseline measurement, not drug-treatment-induced — true causal drug-methylation data does not exist at scale in public databases
- Promoter sequence is identical across cell lines for the same gene — cannot encode cell-specific drug response without additional genomic context
- Single-gene methylation as a scalar covariate ignores genome-wide patterns

## Future Work

- Add CCLE gene expression profiles (RNA-seq) as a fifth input branch
- Add mutation status (WES) for targeted therapy prediction
- Replace Morgan fingerprints with graph neural network or ChemBERTa encoder
- Extend methylation from single-gene scalar to genome-wide 450K array subset
- Contrastive learning to better separate sensitive/resistant phenotypes within tissue type

---

## References

- Yang W et al. Genomics of Drug Sensitivity in Cancer (GDSC). *Nucleic Acids Research*, 2013
- Ghandi M et al. Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature*, 2019
- Liu P et al. tCNNS: Drug Response Prediction via Twin CNN. *Bioinformatics*, 2019
- Angermueller C et al. DeepCpG: prediction of single-cell DNA methylation. *Genome Biology*, 2017
- Liu Y et al. DeepMethylation: methylation prediction with transformer and k-mer encoding. 2023
- Chen T et al. StableDNAm: stable prediction of DNA methylation. *Briefings in Bioinformatics*, 2023
- Hu J et al. Squeeze-and-Excitation Networks. *CVPR*, 2018

---

## Citation

If you use this project or its findings, please cite:

```
@misc{genepromdl2025,
  title   = {GenePromDL: Systematic Investigation of Biological Feature 
             Contribution in Cancer Drug Sensitivity Prediction},
  year    = {2025},
  note    = {Undergraduate Final Year Project, AIML},
}
```
