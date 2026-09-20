# Nirmalya Sarkar — Machine Learning Portfolio

## Who Nirmalya is
Nirmalya Sarkar builds applied machine learning projects end to end: a real dataset, a Jupyter notebook that shows the reasoning, a Flask or full-stack backend, and a frontend designed for the subject. His projects span fraud detection, forecasting, recommendation, simulation, and conversational AI. Stack: Python, Flask, scikit-learn, Node.js. Focus: applied ML, explainability, NLP. Currently exploring: computer vision and medical imaging. GitHub: https://github.com/Nirmalya625/nirmalya-sarkar . LinkedIn: https://www.linkedin.com/in/nirmalyasarkar/

## Projects overview
Eight shipped projects, each with a working app: Fraud Detection Console, SIGNAL stock predictor, World Cup 2026 Predictor, Nova AI companion, Interview Practice Coach, Conversational AI Assistant (RAG), Movie Recommender, and a Healthcare Platform. Every project follows the same format: data, model, backend, app. Everything he has built, made, or created: list of projects, portfolio, work.

## Project: Fraud Detection Console
Supervised learning and explainability. Screens transactions from a 6.36M-row PaySim-style dataset for fraud. Compared Logistic Regression, Random Forest, XGBoost, and Isolation Forest, then shipped a Random Forest pipeline (PR-AUC 0.997) behind a dark "screening desk" interface with per-prediction SHAP explanations and an animated CLEARED / FLAGGED verdict stamp. Tech: Random Forest, XGBoost, Isolation Forest, SHAP, Flask.

## Project: SIGNAL — Stock Predictor
Supervised learning and forecasting. Predicts a next-day buy/sell signal across a multi-asset pipeline that pulls in market, sector, and VIX context. Compares Logistic Regression against Random Forest and outputs a probability with a confidence score rather than a flat call. Tech: Logistic Regression, Random Forest, multi-asset pipeline, Flask.

## Project: World Cup 2026 Predictor
Simulation and probabilistic modeling. Knockout-stage prediction engine trained on roughly 49,500 historical matches. An Elo rating engine feeds a logistic regression win-probability model, and outcomes are resolved with Monte Carlo bracket simulation, presented in an animated stadium-night interface. Tech: Elo ratings, Monte Carlo, scikit-learn, Flask and JavaScript.

## Project: Nova — AI Companion
NLP and applied ML. A browser-based AI companion with long-term memory: SQLite storage retrieved with TF-IDF and a trained logistic regression memory-lookup classifier, VADER-based emotion detection, live weather, web search, voice input, and a procedural audio layer, all served from a Flask backend. Tech: TF-IDF, Logistic Regression, VADER, Groq LLM, SQLite.

## Project: Interview Practice Coach
NLP and sentiment analysis. A mock-interview chatbot that reads tone as you type. VADER sentiment combined with hedging and filler-word heuristics drives a live emotion meter and adaptive follow-up questions, with an LLM-generated closing summary that falls back to a rule-based one offline. Tech: VADER, adaptive questioning, Groq LLM, Flask.

## Project: Conversational AI Assistant
NLP and retrieval-augmented generation (RAG). A streaming chat app with conversation memory and retrieval-augmented generation over uploaded documents, split into dedicated chatbot, memory, and RAG services, deployed live rather than left on localhost. Tech: RAG, Groq, streaming UI, Render deploy.

## Project: Movie Recommender
Unsupervised learning and content-based filtering. Content-based recommendation engine over the TMDB 5000 dataset. Vectorizes genres, keywords, cast, and director with configurable feature weighting, tuned with grid search for precision@5. Tech: CountVectorizer, cosine similarity, scikit-learn.

## Project: Healthcare Platform
Full-stack with OCR and document AI. Connects patients, sample collectors, labs, and admins in one booking-to-report workflow, with AI/OCR extraction of prescriptions. Node.js and Express on the backend, a Python OCR/NLP service, and MySQL/PostgreSQL for storage. Tech: Node.js, Express, OCR, PostgreSQL.

## Techniques and skills
Supervised learning: Logistic Regression, Random Forest, XGBoost, Gradient Boosting. Unsupervised learning: K-Means, Hierarchical Clustering, DBSCAN, Isolation Forest. Model explainability: SHAP, threshold tuning, Precision-Recall AUC. NLP and language systems: TF-IDF retrieval, VADER sentiment, retrieval-augmented generation, LLM integration with Groq. Simulation and probability: Elo rating systems, Monte Carlo simulation, Brier score validation. Deep learning and vision (studied in depth, not yet deployed): CNNs, U-Net, Vision Transformers, Autoencoders. Skills, technologies, tools, what does he know.

## This assistant
The chat assistant on this page runs on the Conversational AI Assistant project: Flask backend, streaming Groq responses, and TF-IDF retrieval over a knowledge base about Nirmalya. Visitors can also upload a PDF, such as a job description, and ask how Nirmalya's projects match it.
