# Job Hunt Agent — Hallucination-Free Resume Optimization

An AI pipeline that parses your resume, matches it against job descriptions using FAISS vector search, and rewrites bullet points — with a learned hallucination detector to ensure every generated claim is grounded in your actual experience.

## Architecture
- **ResumeParser** → extracts a structured FactSheet (JSON) from your PDF
- **FAISS Index** → retrieves top-K matching job descriptions
- **ContrastiveRanker** → pairwise MLP with MC Dropout confidence intervals
- **ControlledOptimizer** → rewrites bullets using only verified facts
- **LearnedHallucinationDetector** → binary MLP that gates outputs on P(hallucination)
- **GroundingEvaluator** → reports Hallucination Rate, Faithfulness, NDCG, MRR

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add your Gemini API key when prompted (or set `GEMINI_API_KEY` in a `.env` file)
3. Set your resume PDF path in Cell 16
4. Run all cells top to bottom

## Requirements
- Python 3.10+
- GPU recommended (set `CUDA_VISIBLE_DEVICES` in Cell 2)
