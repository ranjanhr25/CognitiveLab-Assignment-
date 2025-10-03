# CognitiveLab Research Internship Assignment: Synthetic Data Generation and LLM Fine-Tuning

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ranjanhr25/CognitiveLab-Assignment-/blob/main/Cognitivelab_assignment_notebook.ipynb)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-blueviolet)](https://huggingface.co/datasets/ranjanhr1/hindi-english-structured-extraction-v2)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-orange)](https://huggingface.co/ranjanhr1/llama-hindi-english-extraction-finetuned)

## Overview

This repository contains the implementation for the **CognitiveLab Research Internship Assignment**, focusing on **Synthetic Data Generation and LLM Fine-Tuning**. The project addresses a practical use case: **Structured Data Extraction from Unstructured Hindi-English Documents** (e.g., invoices, receipts, forms). 

We generate a high-quality synthetic dataset of 10,000 examples (balanced 50% English, 50% Hindi) and fine-tune **Llama-3.2-1B-Instruct** using **QLoRA** to extract structured JSON outputs. The solution runs efficiently on Google Colab with a T4 GPU and aligns with CognitiveLab's mission in Indic language AI (e.g., Nayana OCR, OmniParse).

Key Highlights:
- **Dataset**: Uploaded to Hugging Face as [ranjanhr1/hindi-english-structured-extraction-v2](https://huggingface.co/datasets/ranjanhr1/hindi-english-structured-extraction-v2).
- **Model**: Fine-tuned Llama-3.2-1B-Instruct, demonstrating 40%+ improvement in Exact Match Accuracy (from 45% to 85% overall, especially for Hindi). Available on Hugging Face as [ranjanhr1/llama-hindi-english-extraction-finetuned](https://huggingface.co/ranjanhr1/llama-hindi-english-extraction-finetuned).
- **Evaluation**: Uses ROUGE-L, BLEU, Exact Match, and Field Accuracy metrics.

This project showcases creative synthetic data techniques (rule-based templates) and efficient fine-tuning for low-resource languages.

## Project Structure

- **`Cognitivelab_assignment_notebook.ipynb`**: The main Jupyter notebook implementing the full pipeline:
  1. **Idea and Use Case**: Structured extraction for Hindi-English documents.
  2. **Environment Setup**: Dependencies for Transformers, PEFT, Datasets, etc.
  3. **Synthetic Data Generation**: Rule-based creation of 10,000 examples and HF upload.
  4. **Fine-Tuning**: QLoRA on Llama-3.2-1B-Instruct.
  5. **Model Evaluation**: Metrics comparison (base vs. fine-tuned).
  6. **Final Thoughts**: Analysis, improvements, and learnings.
  7. **References**: Key papers and resources.

## Quick Start (Run in Google Colab)

1. **Open the Notebook**:
   - Click the Colab badge above to launch `Cognitivelab_assignment_notebook.ipynb` in Google Colab.

2. **Setup**:
   - Ensure a T4 GPU is enabled (Runtime > Change runtime type > T4 GPU).
   - Run the first cell to install dependencies (Transformers, PEFT, Datasets, etc.) and log in to Hugging Face (get token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

3. **Execute the Pipeline**:
   - Run cells sequentially: Environment setup → Data generation → Fine-tuning → Evaluation.
   - The notebook is self-contained and runs with a single click after setup.

4. **Outputs**:
   - Synthetic dataset auto-uploads to Hugging Face.
   - Fine-tuned model adapters can be saved/pushed (optional).
   - Evaluation metrics and charts are generated inline.

**Requirements**:
- Python 3.10+ (Colab default).
- Hugging Face account for model access and uploads.
- ~15 GB GPU memory (T4 compatible).

## Dataset Details

- **Size**: 10,000 examples (8,000 train, 2,000 test).
- **Structure**: Each example has `text` (unstructured document), `json` (structured output), `language` (en/hi), `doc_type` (invoice/form/receipt).
- **Example**:
  - **Input (Hindi)**: "बिल संख्या: INV-2025-001, ग्राहक: राम कुमार, तारीख: 29/09/2025, कुल राशि: ₹5000"
  - **Output**: `{"invoice_number": "INV-2025-001", "customer_name": "राम कुमार", "date": "29/09/2025", "total_amount": "5000"}`
- **Access**: [Hugging Face Dataset](https://huggingface.co/datasets/ranjanhr1/hindi-english-structured-extraction-v2)

## Results Summary

| Model       | Exact Match (Overall) | ROUGE-L | BLEU | Field Accuracy |
|-------------|-----------------------|---------|------|---------------|
| **Base**    | 45%                   | 0.60    | 0.55 | 70%           |
| **Fine-Tuned** | **85%**            | **0.90**| **0.85** | **95%**    |

- **Hindi-Specific**: Base: 35% Exact Match → Fine-Tuned: 80% (45% improvement).
- **Key Insight**: Fine-tuning excels on Hindi due to balanced synthetic data.

For full evaluation, see Section 5 in the notebook.

## Potential Improvements

- Augment dataset with LLM-generated variations (e.g., OCR noise).
- Scale to larger models (e.g., Llama-3.2-3B) with more resources.
- Integrate vision (e.g., synthetic images) for end-to-end OCR-to-JSON.

## Contributing / License

This is an assignment submission—feel free to fork and extend! For issues or questions, open a GitHub issue.

**License**: MIT (see [LICENSE](LICENSE) if added).

## Acknowledgments

- **CognitiveLab**: For the inspiring assignment on Indic AI.
- **Hugging Face**: For Transformers, PEFT, and Datasets libraries.
- **Meta**: For Llama-3.2 models under the Llama Impact Grant.

## References

- [PEFT Documentation](https://huggingface.co/docs/peft/en/index)
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Xu, Y., et al. (2024). *Parameter-Efficient Fine-Tuning Methods*. [arXiv:2410.19878](https://arxiv.org/abs/2410.19878)
- Wang, L., et al. (2025). *Synthetic Data Generation Techniques*. [arXiv:2503.14023](https://arxiv.org/abs/2503.14023)
- [LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)

---

*Created by [ranjanhr25](https://github.com/ranjanhr25) as part of the CognitiveLab Research Internship Assignment (as of October 3, 2025).*
