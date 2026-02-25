# Anemia Prediction AI 🩺 (React Version)

An advanced clinical diagnostic assistant built with **React**, **Vite**, and **Python Serverless Functions**. This application predicts the probability of anemia based on clinical blood reports.

## 🚀 Architecture
- **Frontend**: React (Vite) + Vanilla CSS (Glassmorphism)
- **Backend (API)**: Python Serverless Functions (hosted on Vercel)
- **ML Engine**: Logistic Regression (99.65% Accuracy)

## 📁 Project Structure
- `/api`: Contains the Python ML engine and serverless handler.
- `/frontend`: The React application UI.
- `vercel.json`: Configuration for deployment as a monorepo.

## 🧪 Quick Start (Local)

### 1. Backend
The backend is designed for Vercel Serverless. To run it locally for development, you can use the Flask `app.py` or the Vercel CLI.

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📦 Deployment to Vercel
This project is pre-configured for Vercel Monorepos.
1. Push this folder to GitHub.
2. In Vercel, create a new project from your repo.
3. Vercel will automatically detect the `vercel.json` and deploy both the API and the React frontend.

## 🛠️ Tech Stack
- **Frontend**: React, Vite
- **Backend**: Python, Vercel Functions
- **ML Libraries**: Scikit-learn, XGBoost, Pandas, NumPy
- **Styling**: Modern CSS3 (Glassmorphism)

---
*Disclaimer: This is an AI-powered tool for informational purposes. Always consult with a qualified medical professional.*
