import json
import requests
from langchain_core.tools import tool
from src.config import SERPAPI_API_KEY, TAVILY_API_KEY


@tool
def search_local_sightseeing(query: str, location: str) -> str:
    """
    Search for local sightseeing activities via SerpAPI Google Local.

    Args:
        query: Search query (e.g. 'popular sightseeing activities things to do')
        location: Destination city (e.g. 'Bangkok, Thailand')

    Returns:
        JSON string with local activity options including name, rating, address.
    """
    params = {
        "engine": "google_local",
        "q": query,
        "location": location,
        "api_key": SERPAPI_API_KEY,
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        activities = []
        for place in data.get("local_results", [])[:10]:
            activities.append({
                "name": place.get("title", "Unknown"),
                "rating": place.get("rating", "N/A"),
                "reviews": place.get("reviews", 0),
                "type": place.get("type", ""),
                "address": place.get("address", ""),
                "hours": place.get("hours", ""),
                "price": place.get("price", ""),
            })

        return json.dumps({"activities": activities, "count": len(activities)})

    except requests.exceptions.Timeout:
        return json.dumps({"error": "SerpAPI request timed out", "activities": []})
    except Exception as exc:
        return json.dumps({"error": str(exc), "activities": []})


@tool
def search_tavily(query: str) -> str:
    """
    Search for top travel attractions and tips via Tavily AI search.

    Args:
        query: Natural-language search query (e.g. 'top attractions in Bangkok Thailand')

    Returns:
        JSON string with search answer and relevant results.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "include_answer": "basic",
        "max_results": 5,
        "search_depth": "basic",
    }

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "content": r.get("content", "")[:300],
                "url": r.get("url", ""),
            })

        return json.dumps({
            "answer": data.get("answer", ""),
            "results": results,
        })

    except requests.exceptions.Timeout:
        return json.dumps({"error": "Tavily request timed out", "results": []})
    except Exception as exc:
        return json.dumps({"error": str(exc), "results": []})
