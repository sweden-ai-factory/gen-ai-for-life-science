# 5. Concluding Remarks

Across the four sessions, we've seen an introduction to generative AI concepts and their practical applications to the life sciences.

## What We Covered

In **Session 1**, we introduced the fundamental concepts underpinning deep learning and built some primitive neural networks.

In **Session 2**, we built a simple image generator using an autoencoder. We compressed image data into a low-dimensional latent space, reconstructed the input image and decoded new latent points into synthetic images. This gave us the first version of a generative model and showed why latent spaces can be useful for interpretation.

In **Session 3**, we moved from image data to sequence data. We introduced protein language models and ubiquitous techniques used in genAI, such as tokenisation, embedding, attention and transfer learning. We used ESM2 as a frozen encoder and trained a smaller model on top to predict a discrete protein property.

Finally, in **Session 4**, we connected sequence modelling to a drug-discovery application. We learned about autoregressive models, vocabularies, positional embeddings and the advantages of using transformers. We built a transformer-based codon predictor, which reconstructed plausible codon sequences from amino acid sequences.

## Other Applications of Generative AI to Life Sciences

The models in this workshop are small and simple teaching examples, but GenAI for LS extends much further.

For example, generative AI is being used in:

- **Inpainting microscopy data:** This involves learning synthetic image segments to remove artefacts, restore missing optical sections and potentially reduce imaging time or photodamage.
- **Vision-language models for medical imaging:** Vision language models (VLMs) are an extension of large language models which produce text data conditioned on image data. In medical imaging, VLMs are used to automatically generate diagnoses and reports in radiology, pathology or microscopy images.
- **Diffusion models for image segmentation with uncertainty:** Diffusion models, which are known primarily for image generation, can be used to create image masks of segmentation candidates. By running these models multiple times, it is possible to track segmentation uncertainty.

As generative models become increasingly capable, we're uncovering new and profound applications to the life sciences. Used carefully, generative AI gives life scientists a practical way to learn from complex data and build interpretable tools that can support real biological and clinical discovery.

## Tips for Using Generative AI in Life Science

- Set random seeds and document everything you do so experiments are easy to reproduce.
- Clean the data before modelling: check labels, identifiers, duplicates, missing values and leakage between train and test splits.
- Make data **FAIR** where possible (findable, accessible, interoperable and reusable). Open-source your data and put it on a repository like [Zenodo](https://zenodo.org/).
- Watch for hallucinations and plausible-looking nonsense, especially when models generate text, images, structures or sequences. Always evaluate your outputs.
- Keep simple baselines. A complex model should beat a simple method for the right reasons.
- Always treat generated outputs as hypotheses until they are validated experimentally or clinically.

## How Sweden AI Factory Can Help

Sweden AI Factory supports start-ups, small-to-medium enterprises, academics, and public sector workers who want to use AI. Our services are fully-subsidised by the Horizon Europe EuroHPC JU initiative, the SRC and Vinnova.

We can help identify opportunities to use AI, design workflows, build prototypes, scale models to high-performance computers (which we also help you access), and evaluate model behaviour.

Sweden AI Factory has its own Life Science team, but we accept projects from **any discipline**. You can learn more and get in touch through the [Sweden AI Factory website](https://swedenaifactory.se/).
