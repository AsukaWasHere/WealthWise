import os
import json
import boto3
import requests
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# --- INITIAL SETUP ---
load_dotenv()
app = FastAPI()

# --- DATABASE SETUP (SQLAlchemy) ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DATABASE MODELS ---
class AnalysisRequest(Base):
    __tablename__ = "analysis_requests"
    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    income = Column(Integer)
    goals = Column(Text)
    current_investments = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    opportunity = Column(Text, nullable=True)
    risk_assessment = Column(Text, nullable=True)
    suggested_action = Column(Text, nullable=True)
    graph_b64 = Column(Text, nullable=True)

# Create the database tables if they don't exist
Base.metadata.create_all(bind=engine)

# Dependency to get a DB session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PYDANTIC MODELS ---
class UserProfile(BaseModel):
    age: int
    income: int
    goals: str
    current_investments: str

# --- DATA LAYER & AI SERVICES ---

def get_historical_market_data(api_key: str):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&outputsize=compact&apikey={api_key if api_key else 'demo'}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("Time Series (Daily)", {})
        if not data:
             raise ValueError("No time series data found in API response.")
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df['4. close'].head(90).to_json()
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return "{'error': 'Could not fetch historical market data.'}"


def get_analysis_and_graph_data_from_nova(profile: UserProfile, historical_data: str):
    try:
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION_NAME")
        )
        prompt = f"""
        You are an expert financial analyst for an application named PolyFox. Your task is to analyze a client's profile and recent historical market data to generate a personalized financial analysis.

        Your response MUST be a single, valid JSON object containing exactly four keys: "opportunity", "risk_assessment", "suggested_action", and "graph_data".
        - The "graph_data" key must be a JSON object itself, with two keys: "title" and "points".

        Client Profile:
        - Age: {profile.age}
        - Annual Income: {profile.income} INR
        - Financial Goals: "{profile.goals}"
        - Current Investments: "{profile.current_investments}"

        Historical Market Data:
        {historical_data}
        """
        request_body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 2048, "temperature": 0.5}
        }
        response = bedrock_client.invoke_model(modelId='amazon.nova-pro-v1:0', body=json.dumps(request_body))
        result = json.loads(response['body'].read())
        reply_text = result.get('content', [{}])[0].get('text', '{}')
        return json.loads(reply_text)
    except Exception as e:
        print(f"Error invoking AWS Nova Pro model: {e}")
        raise HTTPException(status_code=500, detail=f"AI model invocation failed: {e}")


def plot_analysis_graph(graph_data: dict):
    try:
        title, points = graph_data.get('title', 'Financial Projection'), graph_data.get('points', [])
        if not points: return None
        df = pd.DataFrame(points)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.plot(df['month'], df['value'], marker='o', linestyle='-', color='#1B365D')
        ax.set_title(title, fontsize=16, weight='bold', color='#1A202C')
        ax.set_ylabel("Projected Value (INR)", fontsize=12, color='#4A5568')
        ax.tick_params(axis='x', rotation=45)
        for index, row in df.iterrows():
            ax.text(row['month'], row['value'] + (df['value'].max() * 0.02), f"₹{row['value']:,}", ha='center', size=9)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Error plotting graph: {e}")
        return None

# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "PolyFox AI Financial Advisor backend is running."}

@app.post("/api/analyze")
def analyze_profile(profile: UserProfile, db: Session = Depends(get_db)):
    api_key = os.getenv("FINANCIAL_DATA_API_KEY")
    historical_data = get_historical_market_data(api_key)
    ai_analysis = get_analysis_and_graph_data_from_nova(profile, historical_data)
    graph_b64 = plot_analysis_graph(ai_analysis.get("graph_data", {}))

    db_request = AnalysisRequest(
        age=profile.age,
        income=profile.income,
        goals=profile.goals,
        current_investments=profile.current_investments,
        opportunity=ai_analysis.get("opportunity"),
        risk_assessment=ai_analysis.get("risk_assessment"),
        suggested_action=ai_analysis.get("suggested_action"),
        graph_b64=graph_b64
    )
    db.add(db_request)
    db.commit()
    
    full_response = {
        "summary": f"Personalized analysis for a {profile.age}-year-old.",
        **ai_analysis,
        "graph": graph_b64
    }
    
    return {"status": "success", "insight": full_response}
```eof

#### **Step 2: Check Your Render Environment Variables**
To fix the `matplotlib` warning permanently, make sure you have this environment variable set in your Render dashboard under the **"Environment"** tab.

* **Key:** `MPLCONFIGDIR`
* **Value:** `/var/data/matplotlib`

#### **Step 3: Push to GitHub and Redeploy**
1.  Save the corrected `main.py` file.
2.  Commit and push the changes to your GitHub repository.
    ```bash
    git add backend/main.py
    git commit -m "FIX: Add missing os import and correct logic"
    git push
    ```
3.  Render will automatically start a new deployment. This time, the `NameError` will be gone, `matplotlib` will have a place to write its cache, and your application should deploy successfully.`
