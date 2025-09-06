# LegalDocGPT - Legal Document Summarization System

A comprehensive system for simplifying legal documents into plain English summaries using advanced NLP techniques.

## 🎯 Project Overview

LegalDocGPT transforms complex legal documents (contracts, agreements, affidavits, wills, etc.) into structured, easy-to-understand summaries with bullet points and key information extraction.

## ✨ Key Features

- **Multi-format Support**: Handles PDF and DOCX documents
- **Intelligent Summarization**: Uses Flan-T5-small for abstractive summarization
- **Structured Output**: Automatically extracts parties, dates, amounts, and durations
- **Multiple Interfaces**: Web UI (Gradio), API (FastAPI), and command-line tools
- **Evaluation Metrics**: Built-in ROUGE scoring for quality assessment
- **Fine-tuning Support**: Custom model training on legal document datasets

## 📊 Performance Metrics

Our InLegalBERT-enhanced summarization approach achieves:
- **ROUGE-1**: 0.361 (36.1% overlap with reference summaries)
- **ROUGE-2**: 0.127 (12.7% bigram overlap)
- **ROUGE-L**: 0.185 (18.5% longest common subsequence)

### Performance Evolution
| Approach | ROUGE-1 | ROUGE-2 | ROUGE-L | Improvement |
|----------|---------|---------|---------|-------------|
| Baseline | 0.070 | 0.017 | 0.047 | - |
| Enhanced | 0.305 | 0.122 | 0.183 | 4.4x |
| InLegalBERT | 0.342 | 0.124 | 0.177 | 4.9x |
| **Comprehensive InLegalBERT** | **0.361** | **0.127** | **0.185** | **5.2x** |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd LegalDocGPT

# Install dependencies
pip install -r requirements.txt
```

### 2. Web Interface (Gradio)

**Standard Interface:**
```bash
python scripts/gradio_app.py
```
Access at `http://localhost:7860`

**InLegalBERT Enhanced Interface:**
```bash
python scripts/gradio_inlegalbert_app.py
```
Access at `http://localhost:7861`

### 3. API Server

```bash
python api/server.py
```
API available at `http://localhost:8000`

### 4. Command Line Processing

```bash
# Process a single document
python scripts/direct_summarization.py

# Run evaluation
python scripts/eval_rouge.py
```

## 🏗️ Architecture

### Data Pipeline

```
Input Documents (PDF/DOCX)
    ↓
Text Extraction (PyPDF2/python-docx)
    ↓
Preprocessing & Chunking
    ↓
Flan-T5 Summarization
    ↓
Post-processing & Structuring
    ↓
Output (Text + PDF)
```

### Key Components

1. **Text Extraction**: `scripts/extract_text.py`
2. **Dataset Building**: `scripts/build_dataset.py`
3. **Direct Summarization**: `scripts/direct_summarization.py`
4. **Enhanced Processing**: `scripts/enhanced_legal_summarizer.py`
5. **InLegalBERT Enhanced**: `scripts/inlegalbert_enhanced_summarizer.py`
6. **Comprehensive InLegalBERT**: `scripts/inlegalbert_comprehensive_summarizer.py`
7. **Final InLegalBERT**: `scripts/inlegalbert_final_summarizer.py`
8. **Fine-tuning**: `scripts/fine_tune_legal_summarizer.py`
9. **Evaluation**: `scripts/eval_rouge.py`
10. **Web Interfaces**: `scripts/gradio_app.py`, `scripts/gradio_inlegalbert_app.py`

## 📁 Project Structure

```
LegalDocGPT/
├── api/                    # FastAPI server
│   └── server.py
├── data/                   # Data directories
│   ├── input/             # Original documents
│   ├── output/            # Reference summaries
│   ├── predictions_text/  # Generated summaries
│   └── predictions_pdf/   # Generated PDFs
├── dataset/               # Training datasets
│   ├── dataset.jsonl
│   └── dataset.csv
├── models/                # Fine-tuned models
│   └── legal_summarizer/
├── scripts/               # Processing scripts
│   ├── extract_text.py
│   ├── build_dataset.py
│   ├── direct_summarization.py
│   ├── enhanced_legal_summarizer.py
│   ├── fine_tune_legal_summarizer.py
│   ├── eval_rouge.py
│   └── gradio_app.py
├── legaldoc-ui/           # Next.js frontend
└── requirements.txt
```

## 🔧 Usage Examples

### Web Interface

1. Open `http://localhost:7860`
2. Paste legal document text or use the sample
3. Click "Summarize Document"
4. View structured summary with extracted information

### API Usage

```python
import requests

# Upload and process document
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/process', files=files)
    result = response.json()

print(result['points'])  # List of summary points
```

### Command Line

```bash
# Process all documents in dataset
python scripts/enhanced_legal_summarizer.py

# Evaluate results
python scripts/eval_rouge.py

# Fine-tune model
python scripts/fine_tune_legal_summarizer.py
```

## 📈 Evaluation Results

### Baseline vs Enhanced Approach

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| ROUGE-1 | 0.070 | 0.305 | 4.4x |
| ROUGE-2 | 0.017 | 0.122 | 7.2x |
| ROUGE-L | 0.047 | 0.183 | 3.9x |

### Sample Output

**Input**: Complex legal agreement text

**Output**:
```
MUTUAL NON-DISCLOSURE AGREEMENT - Legal Summary (InLegalBERT Enhanced)

📅 Date: 22nd day of August, 2025
👥 Parties:
   • InnovateNext Technologies Pvt. Ltd.
   • DataWise Analytics LLP

💰 Key Amounts:
   • ₹20,00,000
   • ₹10,00,000

⚖️ Legal Entities:
   • Partnership
   • Clause 2
   • Agreement
   • Mumbai
   • Companies Act, 2013

📋 Legal Summary:
1. This Agreement was signed on 22 August 2025.
2. Both parties agree to protect confidential information.
3. The agreement is effective for 3 years.
4. Information must not be disclosed to third parties.
5. Governing law is Indian law with Mumbai jurisdiction.
```

## 🛠️ Advanced Features

### Fine-tuning

Train custom models on your legal document dataset:

```bash
python scripts/fine_tune_legal_summarizer.py
```

### Hybrid Approaches

- **Extractive + Abstractive**: Combine sentence extraction with rewriting
- **InLegalBERT Integration**: Use domain-specific legal embeddings trained on Indian legal documents
- **Multi-model Ensemble**: Combine multiple summarization approaches
- **Legal Entity Recognition**: Automatic extraction of legal entities, clauses, and provisions

### Post-processing Rules

- Automatic party extraction
- Date and amount identification
- Duration and termination clause detection
- Structured formatting with emojis and categories

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure transformers and torch are properly installed
2. **PDF Generation Issues**: Check fpdf2 installation and font availability
3. **Memory Issues**: Reduce batch size or use smaller models
4. **ROUGE Score Errors**: Install rouge-score package

### Performance Optimization

- Use GPU acceleration for faster processing
- Implement caching for repeated documents
- Optimize chunk sizes for your document types
- Consider model quantization for deployment

## 📚 Dependencies

### Core Requirements
- Python 3.8+
- transformers >= 4.35.2
- torch >= 2.1.1
- fastapi >= 0.104.1
- gradio >= 5.44.1
- fpdf2 >= 2.7.6
- rouge-score >= 0.1.2

### Optional Dependencies
- datasets (for fine-tuning)
- accelerate (for training)
- spacy (for NER)
- scikit-learn (for TF-IDF)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- Flan-T5 model by Google
- [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT) for Indian legal domain-specific embeddings
- ROUGE evaluation metrics
- Indian Institute of Technology, Kharagpur for InLegalBERT research

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

---

**LegalDocGPT** - Making legal documents accessible to everyone! 📋⚖️
