# 4. Drug Discovery Application - Translating Proteins Back to DNA

:::{objectives}
- Explain how sequence-to-sequence models and codon degeneracy motivate protein-to-DNA prediction for mRNA vaccine design
- Build a transformer prediction head with positional embeddings on top of frozen ESM2 features
- Apply codon validity constraints so predicted DNA sequences encode the correct amino acids
- Evaluate the codon predictor against a highest-frequency-codon baseline
:::
