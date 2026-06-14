import json
import requests
from langchain_core.tools import tool
from src.config import SERPAPI_API_KEY


@tool
def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    return_date: str,
    adults: int = 1,
) -> str:
    """
    Search round-trip flights via SerpAPI Google Flights.

    Args:
        departure_id: IATA airport code for origin (e.g. 'DEL' for Delhi)
        arrival_id: IATA airport code for destination (e.g. 'BKK' for Bangkok)
        outbound_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format
        adults: Number of adult passengers

    Returns:
        JSON string with flight options including airline, times, duration, stops, price in INR.
    """
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "return_date": return_date,
        "adults": adults,
        "currency": "INR",
        "type": "1",   # 1 = round trip
        "api_key": SERPAPI_API_KEY,
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        flights = []
        for section in ("best_flights", "other_flights"):
            for group in data.get(section, []):
                legs = group.get("flights", [])
                if not legs:
                    continue
                first_leg = legs[0]
                last_leg = legs[-1]
                flights.append({
                    "airline": first_leg.get("airline", "Unknown"),
                    "flight_number": first_leg.get("flight_number", ""),
                    "departure": first_leg.get("departure_airport", {}).get("time", ""),
                    "arrival": last_leg.get("arrival_airport", {}).get("time", ""),
                    "total_duration_min": group.get("total_duration", 0),
                    "stops": len(legs) - 1,
                    "price_INR": group.get("price", "N/A"),
                    "carbon_emissions_g": group.get("carbon_emissions", {}).get("this_flight", "N/A"),
                })
                if len(flights) >= 6:
                    break
            if len(flights) >= 6:
                break

        return json.dumps({"flights": flights, "count": len(flights)})

    except requests.exceptions.Timeout:
        return json.dumps({"error": "SerpAPI request timed out", "flights": []})
    except Exception as exc:
        return json.dumps({"error": str(exc), "flights": []})
