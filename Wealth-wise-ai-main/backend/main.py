import os
import json
import boto3
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import io
import base64
import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- INITIAL SETUP ---
load_dotenv()
app = FastAPI()

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

def create_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            id SERIAL PRIMARY KEY,
            age INTEGER,
            income NUMERIC,
            current_savings NUMERIC,
            investment_experience TEXT,
            monthly_budget NUMERIC,
            goals TEXT[],
            investment_timeline INTEGER,
            risk_tolerance TEXT,
            risk_allocation JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

create_table()


# --- PYDANTIC MODELS ---
class UserProfile(BaseModel):
    age: int
    income: float
    current_savings: float
    investment_experience: str
    monthly_budget: float
    goals: list[str]
    investment_timeline: int
    risk_tolerance: str

# --- RISK PROFILE FUNCTIONS ---
def calculate_allocation(age, income, current_savings, investment_exp, monthly_budget, goals, timeline, risk_tolerance):
    allocation = {"Bonds": 0, "REITs": 0, "Stocks": 0, "Growth Funds": 0, "Crypto": 0}
    age_factor = 1.2 if age < 30 else 1.0 if age < 45 else 0.8 if age < 60 else 0.5
    risk_multipliers = {
        "Conservative": 0.5, "Moderate Conservative": 0.7, "Moderate": 1.0,
        "Moderate Aggressive": 1.2, "Aggressive": 1.5
    }
    risk_factor = risk_multipliers.get(risk_tolerance, 1.0)
    goal_weights = {
        "retirement planning": {"Bonds": 30, "REITs": 10, "Stocks": 25, "Growth Funds": 25, "Crypto": 10},
        "home purchase": {"Bonds": 40, "REITs": 10, "Stocks": 30, "Growth Funds": 15, "Crypto": 5},
        "education funding": {"Bonds": 20, "REITs": 10, "Stocks": 35, "Growth Funds": 25, "Crypto": 10},
        "wealth building": {"Bonds": 15, "REITs": 10, "Stocks": 40, "Growth Funds": 25, "Crypto": 10},
        "emergency fund": {"Bonds": 50, "REITs": 10, "Stocks": 20, "Growth Funds": 15, "Crypto": 5}
    }

    for goal in goals:
        g = goal.lower().replace(" ", "_")
        if "retirement" in g: g = "retirement_planning"
        if "home" in g: g = "home_purchase"
        if "education" in g: g = "education_funding"
        if "wealth" in g: g = "wealth_building"
        if "emergency" in g: g = "emergency_fund"
        
        if g in goal_weights:
            for k in allocation:
                allocation[k] += goal_weights[g][k]

    if goals:
        for k in allocation:
            allocation[k] /= len(goals)

    for k in allocation:
        allocation[k] *= age_factor * risk_factor

    total = sum(allocation.values())
    if total == 0: return {key: 0 for key in allocation}
    
    for k in allocation:
        allocation[k] = round(allocation[k] / total * 100, 1)
    return allocation

def plot_risk_profile_base64(allocation):
    import numpy as np
    categories, values = list(allocation.keys()), list(allocation.values())
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(values) + 10)
    ax.set_title("Risk Profile Visualization", size=15, weight='bold', y=1.1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "WealthWise AI Backend is running."}

@app.post("/api/analyze")
def analyze_profile(profile: UserProfile):
    try:
        risk_allocation = calculate_allocation(**profile.dict())
        graph_b64 = plot_risk_profile_base64(risk_allocation)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_profiles (age, income, current_savings, investment_experience, monthly_budget, goals, investment_timeline, risk_tolerance, risk_allocation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (profile.age, profile.income, profile.current_savings, profile.investment_experience,
             profile.monthly_budget, profile.goals, profile.investment_timeline, profile.risk_tolerance,
             json.dumps(risk_allocation))
        )
        conn.commit()
        cur.close()
        conn.close()

        return {
            "status": "success",
            "insight": {
                "summary": f"Personalized analysis for a {profile.age}-year-old with a '{profile.risk_tolerance}' risk tolerance.",
                "risk_allocation": risk_allocation,
                "graph": graph_b64
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
