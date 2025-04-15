from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from langchain_fireworks import Fireworks

# ===== Logger Setup =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== FastAPI App Setup =====
app = FastAPI()

# ===== Fireworks LLM Initialization =====
placement_llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-405b-instruct",
    base_url="https://api.fireworks.ai/inference/v1/completions",
    temperature=0.3
)

# ===== Request Schema =====
class PlacementChatbotRequest(BaseModel):
    query: str  # User's placement-related query

# ===== Helper: Clean Response into Bullet Points =====
def clean_response(response: str) -> str:
    cleaned = response.replace("\n", " ").replace("*", "").strip()
    bullet_points = [f"- {point.strip()}" for point in cleaned.split(". ") if point and len(point.strip()) > 2]

    # Ensure exactly 5 bullet points
    if len(bullet_points) > 5:
        bullet_points = bullet_points[:5]
    elif len(bullet_points) < 5:
        bullet_points += ["- (Additional relevant point needed)"] * (5 - len(bullet_points))

    return "\n".join(bullet_points)

# ===== Placement Guidance Function =====
def generate_placement_response(query: str) -> dict:
    prompt = f"""
    Placement Guidance Chatbot:
    User Query: {query}

    Provide a response with EXACTLY 5 clear bullet points:
    - Relevant to campus placements, interviews, resume building, or job search.
    - Accurate and helpful.
    - Skip anything unrelated to placements.
    - If irrelevant, respond with: "I only provide placement-related guidance."
    """

    try:
        response = placement_llm.invoke(prompt, max_tokens=500)
        logger.info(f"LLM Response: {response}")

        if isinstance(response, str):
            cleaned = clean_response(response)
        elif isinstance(response, dict) and "text" in response:
            cleaned = clean_response(response["text"])
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format from LLM.")

        return {"response": cleaned}

    except Exception as e:
        logger.error(f"Failed to generate placement response: {e}")
        raise HTTPException(status_code=500, detail="Error generating chatbot response.")

# ===== FastAPI Endpoint =====
@app.post("/placement_chatbot")
async def placement_chatbot(request: PlacementChatbotRequest):
    logger.info(f"Received query: {request.query}")
    result = generate_placement_response(request.query)
    return {
        "conversation": [
            {"role": "user", "message": request.query},
            {"role": "bot", "message": result['response']}
        ]
    }
