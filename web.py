import requests
import json
import time

WEBHOOK_URL = "https://projectu.app.n8n.cloud/webhook-test/agri-intel-chat"
HEADERS = {"Content-Type": "application/json"}


def send_query(message, session_id="test-session"):
    payload = {
        "message": message,
        "sessionId": session_id
    }

    print("\n===============================")
    print(f"📤 Sending Query: {message}")
    print("===============================\n")

    response = requests.post(WEBHOOK_URL, json=payload, headers=HEADERS)

    try:
        data = response.json()
        print("📥 Response:")
        print(json.dumps(data, indent=2))
    except Exception:
        print("❌ Error: Response not JSON")
        print(response.text)

    time.sleep(1)  # avoid rate-limiting


def main():
    print("🔥 Testing Agri Intel Webhook with Real Agricultural Queries...\n")

    queries = [
        # Crop yield prediction
        "Predict the crop yield for paddy if rainfall is 320mm, soil is loamy, NPK 40-25-20, pH 6.4, temperature 28°C, irrigation moderate.",

        # Fertilizer recommendation
        "Suggest fertilizers for wheat crop. Soil nutrients are N=50, P=20, K=30 and pH is 6.5. Soil is sandy loam.",

        # Crop recommendation
        "My soil NPK values are 30-15-20, pH is 7.0, rainfall around 280mm, temperature 30°C. What crops should I grow this season?",

        # Market price prediction
        "Predict tomato price next week for Bangalore city market.",

        # Pest prediction
        "There is high humidity and rainfall in my cotton field. Any pest risk expected?",

        # General agriculture advice
        "What is the best irrigation method for sugarcane farming?",

        # Disease related query (no image needed for testing)
        "My tomato leaves have yellow patches and curling. What disease could this be?",

        # Soil improvement
        "How can I improve low nitrogen levels in my soil naturally?",

        # Weather impact
        "How will sudden heavy rainfall affect groundnut crop?",

        # Organic farming question
        "Give organic pest control suggestions for brinjal (eggplant)."
    ]

    for q in queries:
        send_query(q)


if __name__ == "__main__":
    main()
