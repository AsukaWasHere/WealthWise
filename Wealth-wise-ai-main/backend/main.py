# 1. Standard library imports
import os
import time

# 2. Third-party imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 3. Load environment variables from your .env file
# This should be one of the first things your application does.
load_dotenv()

# 4. Create the FastAPI app instance
# The variable MUST be named 'app' for the start command 'main:app' to work.
app = FastAPI()

# 5. Configure CORS Middleware
# This allows your frontend to make requests to this backend.
origins = [
    "http://localhost:3000",
    # TODO: Add your Vercel frontend URL here once it's deployed.
    # e.g., "https://your-project-name.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Pydantic Model for incoming data validation
class UserProfile(BaseModel):
    age: int
    income: int
    goals: str
    current_investments: str

# 7. API Endpoints (Routes)
@app.get("/")
def read_root():
    """ A simple 'health check' endpoint to confirm the server is running. """
    return {"status": "ok", "message": "AI Financial Advisor backend is running."}


@app.post("/api/analyze")
def analyze_profile(profile: UserProfile):
    """
    This is the main endpoint that receives the user profile, simulates
    AI analysis, and returns a structured insight.
    """
    print(f"Received profile for analysis: {profile.dict()}")

    # --- This is where your real application logic will go ---
    # TODO: 1. Connect to your database using os.getenv("DATABASE_URL")
    # TODO: 2. Fetch real-time data from a News API using os.getenv("NEWS_API_KEY")
    # TODO: 3. Construct a detailed prompt for your AI model
    # TODO: 4. Call the Generative AI API using os.getenv("GEMINI_API_KEY")
    # TODO: 5. Save the AI's response to your database

    # For now, we simulate a delay and return a structured mock response
    time.sleep(2)

    mock_insight = {
        "summary": f"Analysis for a {profile.age}-year-old with a goal of '{profile.goals}'",
        "opportunity": "Given recent positive sentiment in the renewable energy sector, a thematic mutual fund could align with your long-term growth objectives.",
        "risk_assessment": "Your portfolio shows a high concentration in technology stocks. We recommend diversifying into other sectors like FMCG or healthcare to mitigate sector-specific risks.",
        "suggested_action": "Research the top 3 renewable energy funds in India. Compare their expense ratios and past 5-year performance."
    }

    return {"status": "success", "insight": mock_insight}
