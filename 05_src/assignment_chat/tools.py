from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import requests
from pydantic import BaseModel, ConfigDict, Field


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class WeatherArguments(ToolArguments):
    pass


class ActivityArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=2000)


class BudgetArguments(ToolArguments):
    budget: float = Field(ge=0, description="Available budget for the entire group, CAD.")
    transport: float = Field(ge=0, description="Transport cost per person, CAD.")
    food: float = Field(ge=0, description="Food cost per person, CAD.")
    activities: float = Field(ge=0, description="Activity cost per person, CAD.")
    people: int = Field(ge=1, description="Number of people sharing this budget.")


class CurrentWeather(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    time: str
    temperature_2m: float
    apparent_temperature: float
    precipitation: float = Field(ge=0)
    weather_code: int
    wind_speed_10m: float = Field(ge=0)


class WeatherResponse(BaseModel):
    timezone: str
    current: CurrentWeather


_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def get_toronto_weather() -> dict[str, object]:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": 43.6532,
            "longitude": -79.3832,
            "current": (
                "temperature_2m,apparent_temperature,precipitation,"
                "weather_code,wind_speed_10m"
            ),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "timezone": "America/Toronto",
        },
        timeout=15,
    )
    response.raise_for_status()
    weather = WeatherResponse.model_validate(response.json())
    current = weather.current
    datetime.fromisoformat(current.time)
    if current.weather_code not in _WEATHER_CODES:
        raise ValueError("Open-Meteo returned an unsupported weather code.")
    return {
        "location": "Toronto, Ontario",
        "time": current.time,
        "timezone": weather.timezone,
        "conditions": _WEATHER_CODES[current.weather_code],
        "temperature_c": current.temperature_2m,
        "feels_like_c": current.apparent_temperature,
        "precipitation_mm": current.precipitation,
        "wind_kmh": current.wind_speed_10m,
        "source": "https://open-meteo.com/",
        "limitation": "Current model-based conditions, not an all-day forecast.",
    }


def calculate_budget(
    budget: float, transport: float, food: float, activities: float, people: int
) -> dict[str, object]:
    args = BudgetArguments(
        budget=budget, transport=transport, food=food, activities=activities, people=people
    )
    cents = Decimal("0.01")

    def money(value: float) -> Decimal:
        return Decimal(str(value)).quantize(cents, rounding=ROUND_HALF_UP)

    per_person = money(args.transport) + money(args.food) + money(args.activities)
    total = per_person * args.people
    remaining = money(args.budget) - total
    return {
        "currency": "CAD",
        "people": args.people,
        "budget": float(money(args.budget)),
        "cost_per_person": float(per_person),
        "total_cost": float(total),
        "remaining": float(remaining),
        "within_budget": remaining >= 0,
    }
