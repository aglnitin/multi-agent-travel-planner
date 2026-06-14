import json
import requests
from langchain_core.tools import tool
from src.config import SERPAPI_API_KEY


@tool
def search_hotels(
    destination: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    hotel_class: str = "",
) -> str:
    """
    Search hotels via SerpAPI Google Hotels.

    Args:
        destination: City name to search hotels in (e.g. 'Bangkok')
        check_in_date: Check-in date in YYYY-MM-DD format
        check_out_date: Check-out date in YYYY-MM-DD format
        adults: Number of guests
        hotel_class: Minimum star rating as string '3', '4', or '5' (optional)

    Returns:
        JSON string with hotel options including name, rating, price per night, total cost.
    """
    params = {
        "engine": "google_hotels",
        "q": f"hotels in {destination}",
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "currency": "INR",
        "api_key": SERPAPI_API_KEY,
    }
    if hotel_class:
        params["hotel_class"] = hotel_class

    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hotels = []
        for prop in data.get("properties", [])[:7]:
            rate_info = prop.get("rate_per_night", {})
            total_info = prop.get("total_rate", {})
            hotels.append({
                "name": prop.get("name", "Unknown"),
                "rating": prop.get("overall_rating", "N/A"),
                "reviews": prop.get("reviews", 0),
                "stars": prop.get("hotel_class", ""),
                "location": prop.get("neighborhood", ""),
                "price_per_night_INR": rate_info.get("lowest", "N/A"),
                "total_stay_INR": total_info.get("lowest", "N/A"),
                "amenities": prop.get("amenities", [])[:5],
            })

        return json.dumps({"hotels": hotels, "count": len(hotels)})

    except requests.exceptions.Timeout:
        return json.dumps({"error": "SerpAPI request timed out", "hotels": []})
    except Exception as exc:
        return json.dumps({"error": str(exc), "hotels": []})
