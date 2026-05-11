"""Weather tool — fetches current weather for a city using Open-Meteo (no key)."""

import requests


# Open-Meteo's free geocoding endpoint — turns "Tokyo" into lat/lon
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Open-Meteo's forecast endpoint — current weather at given coordinates
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# WMO weather codes (https://open-meteo.com/en/docs#weathervariables)
# Subset of common conditions; full list has 30+ codes.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: City name, optionally with country (e.g., "Tokyo" or "Paris, FR").
    
    Returns:
        A human-readable weather summary, or an error message string.
    """
    try:
        # Step 1: geocode the city to lat/lon
        geo_response = requests.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en"},
            timeout=10,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return f"Weather error: city '{city}' not found"
        
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        resolved_name = f"{location['name']}, {location.get('country', '?')}"
        
        # Step 2: fetch current weather at those coordinates
        forecast_response = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
                "temperature_unit": "celsius",
            },
            timeout=10,
        )
        forecast_response.raise_for_status()
        current = forecast_response.json()["current"]
        
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind = current["wind_speed_10m"]
        condition = WEATHER_CODES.get(current["weather_code"], "unknown conditions")
        
        return (
            f"{resolved_name}: {temp}°C, {condition}, "
            f"humidity {humidity}%, wind {wind} km/h"
        )
        
    except requests.exceptions.RequestException as e:
        return f"Weather error: network failure: {e}"
    except (KeyError, ValueError, TypeError) as e:
        return f"Weather error: malformed API response: {e}"
    except Exception as e:
        return f"Weather error: {type(e).__name__}: {e}"