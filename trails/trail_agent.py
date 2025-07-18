"""
This script is a trail agent that uses the Judgment framework to generate a travel plan based on the user's preferences and location.

It uses the following tools:
- OpenAI API to generate a travel plan
- Judgment framework to evaluate the travel plan
- Tavily API to search for travel information
- OpenWeatherMap API to get weather information (for weather forecast)
- DistanceMatrixFast API to get distance and duration information (for travel distance and duration)

It uses the following scorers:
- AnswerRelevancyScorer to evaluate the relevance of the travel plan
- ItineraryStructureScorer to evaluate the structure of the travel plan
- PreferenceMatchScorer to evaluate the match between the user's preferences and the travel plan
- FaithfulnessScorer to evaluate the faithfulness of the travel plan

It uses the following models:
- GPT-4o to generate the travel plan

"""

import ast
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import openai
import pandas as pd
import requests
from chromadb.utils import embedding_functions
from custom_scorers import ItineraryStructureScorer, PreferenceMatchScorer
from dotenv import load_dotenv
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer
from judgeval.tracer import Tracer, wrap
from tavily import TavilyClient
from tqdm import tqdm

client = wrap(openai.Client(api_key=os.getenv("OPENAI_API_KEY")))
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    project_name="travel_agent_demo",
)
judgement_client = JudgmentClient(
    api_key=os.getenv("JUDGMENT_API_KEY"),
)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DISTANCEMATRIXFAST_API_KEY = os.getenv("DISTANCEMATRIXFAST_API_KEY")


def initialize_vector_db(trail_data: str) -> chromadb.Collection:
    """
    Initialize ChromaDB with OpenAI embeddings.

    Args:
        trail_data (str): The trail data to populate the vector DB.

    Returns:
        chromadb.Collection: The collection object
    """
    Path("./trails/custom_data/chroma_db").mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path="./trails/custom_data/chroma_db")

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    collection = client.get_or_create_collection(
        name="travel_information", embedding_function=embedding_fn
    )

    populate_vector_db(collection=collection, trail_data=trail_data)

    return collection


def populate_vector_db(
    collection: chromadb.Collection,
    trail_data: str,
) -> None:
    """
    Populate the vector DB with travel information.

    Args:
        collection (chromadb.Collection): The collection to populate
        trail_data (str): The trail data to populate the vector DB with

    trail_data looks like this:
    {
        "name": "Trail Name",
        "description": {
            "city_name": "City Name",
            "state_name": "State Name",
            "area_name": "Area Name",
        }
    }

    Returns:
        None
    """
    # Convert the JSON string to a list of dictionaries
    trail_data = json.loads(trail_data)

    for data in tqdm(trail_data, desc="Populating vector DB"):
        # Convert description dictionary to string for ChromaDB
        description_str = json.dumps(data["description"])

        collection.add(
            documents=[description_str],
            metadatas=[
                {
                    "name": data["name"],
                    "city_name": data["city_name"],
                    "state_name": data["state_name"],
                    "lat": data["lat"],
                    "lng": data["lng"],
                    "features": ", ".join(data["features"]),
                    "activities": ", ".join(data["activities"]),
                }
            ],
            ids=[f"trail_{data['name'].lower().replace(' ', '_')}"],
        )


@judgment.observe(span_type="retriever")
async def query_vector_db(
    collection: chromadb.Collection,
    source: str,
    filter_input: List[str],
    filter_pref: str,
    difficulty: str,
) -> Dict[str, Any]:
    """
    Query the vector database for trails matching activities and features, then sort by distance.

    Args:
        collection (chromadb.Collection): The collection to query
        source (str): The source location
        filter_input (List[str]): The activities and features to filter by
        filter_pref (str): The preference to filter by
        difficulty (str): The difficulty level to filter by

    Returns:
        Dict[str, Any]: The results of the query
    """
    try:
        # Build the query
        query_txt = f"{filter_pref}: {', '.join(filter_input)}"

        # First, find trails matching activities and features (get more than 5 to allow for distance filtering)
        results = collection.query(
            query_texts=[query_txt],
            n_results=50,
            where_document={
                "$or": [
                    {"$contains": ",".join(filter_input)},
                    {"$contains": difficulty},
                ]
            },
        )

        if not results["documents"][0]:
            return {"results": results, "query": query_txt}

        filtered_results = await filter_results(
            source=source,
            results=results,
        )

        # print(filtered_results)
        # exit(0)

        return {"results": filtered_results, "query": query_txt}

    except Exception as e:
        print(f"Error in query_vector_db: {e}")
        return {
            "results": {
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]],
                "distances": [[]],
                "durations": [[]],
            },
            "query": query_txt,
        }


async def filter_results(
    source: str,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Filter the results to only include trails that are within 50 km of the source location.

    Args:
        source (str): The source location
        results (Dict[str, Any]): The results of the query

    Returns:
        Dict[str, Any]: The filtered results
    """
    source_lat, source_lng = await get_lat_lng(source)

    # Calculate distance for each matching trail and add to results
    trails_with_distance = []
    for i, metadata in enumerate(results["metadatas"][0]):
        try:
            trail_lat = float(metadata["lat"])
            trail_lng = float(metadata["lng"])
            distance, duration = await get_distance(
                slat=source_lat,
                slng=source_lng,
                dlat=trail_lat,
                dlng=trail_lng,
            )

            if distance is not None and duration is not None and distance > 0:
                # print(source_lat, source_lng, trail_lat, trail_lng, distance, duration)
                trails_with_distance.append(
                    {
                        "document": results["documents"][0][i],
                        "metadata": metadata,
                        "distance": distance,
                        "duration": duration,
                        "id": results["ids"][0][i],
                    }
                )
            else:
                continue

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error calculating distance for trail: {e}")
            continue

    # Sort by distance
    trails_with_distance.sort(key=lambda x: x["distance"])

    # Take top 5
    top_5_trails = trails_with_distance[:5]

    # Format results back into ChromaDB format
    filtered_results = {
        "documents": [[t["document"] for t in top_5_trails]],
        "metadatas": [[t["metadata"] for t in top_5_trails]],
        "ids": [[t["id"] for t in top_5_trails]],
        "distances": [[t["distance"] for t in top_5_trails]],
        "durations": [[t["duration"] for t in top_5_trails]],
    }

    return filtered_results


@judgment.observe(span_type="search_tool")
def search_tavily(query: str) -> dict:
    """
    Fetch travel data using Tavily API.

    Args:
        query (str): The query to search for

    Returns:
        dict: The results of the search
    """
    API_KEY = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=API_KEY)
    results = client.search(query, num_results=3)
    return results


@judgment.observe(span_type="tool")
async def get_weather(destination: str) -> str:
    """
    Search for weather information.

    Args:
        destination (str): The destination to search for

    Returns:
        str: The weather information
    """
    prompt = f"Weather forecast for {destination}"
    weather_search = search_tavily(prompt)
    example = Example(input=prompt, actual_output=str(weather_search["results"]))
    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        example=example,
        model="gpt-4o",
    )
    return weather_search


@judgment.observe(span_type="tool")
async def get_lat_lng(source: str) -> str:
    """Get the latitude and longitude of a location."""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={source}&limit=1&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url, timeout=5)
    data = response.json()
    lat_lng_search = (data[0]["lat"], data[0]["lon"])
    return lat_lng_search


@judgment.observe(span_type="tool")
async def get_distance(
    slat: float,
    slng: float,
    dlat: float,
    dlng: float,
) -> tuple[float, str]:
    """
    Get the distance and duration between two points.

    Args:
        slat (float): The latitude of the source location
        slng (float): The longitude of the source location
        dlat (float): The latitude of the destination location
        dlng (float): The longitude of the destination location

    Returns:
        tuple[float, str]: The distance and duration
    """
    url = f"https://api-v2.distancematrix.ai/maps/api/distancematrix/json?origins={slat},{slng}&destinations={dlat},{dlng}&key={DISTANCEMATRIXFAST_API_KEY}"
    response = requests.get(url, timeout=8)
    data = response.json()
    prompt = f"Distance and duration between {slat},{slng} and {dlat},{dlng}"
    example = Example(input=prompt, actual_output=str(data))
    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        example=example,
        model="gpt-4o",
    )
    if data["rows"][0]["elements"][0]["status"] == "OK":
        distance = data["rows"][0]["elements"][0]["distance"]["value"]
        duration = data["rows"][0]["elements"][0]["duration"]["text"]
        return distance, duration
    else:
        return None, "Unknown"


@judgment.observe(span_type="Research")
async def research_destination(
    source: str,
    trail_data: str,
    filter_input: List[str],
    filter_pref: str,
    difficulty: str,
) -> str:
    """
    Gather trail information and weather data for creating a travel plan.

    Args:
        source (str): The source location
        trail_data (str): The trail data to populate the vector DB
        filter_input (List[str]): The activities and features to filter by
        filter_pref (str): The preference to filter by
        difficulty (str): The difficulty level to filter by

    Returns:
        str: The travel plan
    """
    # First, check the vector database
    collection = initialize_vector_db(trail_data)
    existing_info = await query_vector_db(
        collection,
        source,
        filter_input,
        filter_pref,
        difficulty,
    )

    # *** NEW: Use the PreferenceMatchScorer ***
    if existing_info and existing_info["results"]["metadatas"][0]:
        retrieval_context = [
            json.dumps(metadata, ensure_ascii=False)
            for metadata in existing_info["results"]["metadatas"][0]
        ]
        eval_example = Example(
            input={filter_pref: filter_input},
            retrieval_context=retrieval_context,
        )

        # Run the evaluation
        judgement_client.run_evaluation(
            scorers=[PreferenceMatchScorer(threshold=0.5)],  # Expect a perfect match
            examples=[eval_example],
            eval_run_name="PreferenceMatchScorer",
            # override=True,
            # model="gpt-4o",
            project_name="travel_agent_demo",
        )

    # Get weather for each of the top 5 trails
    trail_weather = {}
    if (
        existing_info
        and "results" in existing_info
        and existing_info["results"]["metadatas"]
        and existing_info["results"]["metadatas"][0]
    ):
        for metadata in existing_info["results"]["metadatas"][0]:
            city_name = metadata.get("city_name")
            if city_name and city_name not in trail_weather:
                trail_weather[city_name] = await get_weather(city_name)

    tavily_data = {"trail_weather": trail_weather}

    return {"vector_db_results": existing_info, **tavily_data}


@judgment.observe(span_type="function")
async def create_travel_plan(source: str, research_data: dict) -> str:
    """
    Generate a travel itinerary using the researched data.

    Args:
        source (str): The source location
        research_data (dict): The researched data

    Returns:
        str: The travel plan
    """
    vector_db_results = research_data["vector_db_results"]

    # Format trail information with distances
    trail_info = []
    if (
        vector_db_results
        and "results" in vector_db_results
        and vector_db_results["results"]["documents"][0]
    ):
        for i, doc in enumerate(vector_db_results["results"]["documents"][0]):
            metadata = vector_db_results["results"]["metadatas"][0][i]
            distance = (
                vector_db_results["results"]["distances"][0][i]
                if "distances" in vector_db_results["results"]
                else "unknown"
            )
            duration = (
                vector_db_results["results"]["durations"][0][i]
                if "durations" in vector_db_results["results"]
                else "unknown"
            )
            trail_name = metadata.get("name", "Unknown Trail")
            city_name = metadata.get("city_name", "Unknown City")

            # Add weather info if available
            weather_info = ""
            if city_name in research_data.get("trail_weather", {}):
                weather_info = f"\nWeather in {city_name}: {research_data['trail_weather'][city_name]}"

            trail_info.append(
                f"Trail: {trail_name} - Distance from {source}: {distance:.1f} km - Duration to reach: {duration} - {doc}{weather_info}"
            )

    vector_db_context = (
        "\n".join(trail_info) if trail_info else "No matching trails found."
    )

    prompt = f"""
    Create a structured travel itinerary for a trip from {source}.
    
    Matching trails near your location (sorted by distance):
    {vector_db_context}
    
    Focus on the closest trails that match the requested activities and features.
    
    Clearly state and summarize these details in the itinerary:
    - Weather conditions
    - Travel distance and duration
    - Trail length and difficulty
    - Trail activities and features
    
    Summarize the features and activities of each trail in a friendly manner.
    Summarize the weather conditions at each trail location and the best time to visit each trail or start the day when making recommendations.
    """

    response = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert travel planner. Create a detailed plan focusing on trails that match the user's preferences and are closest to their location. Take into account weather conditions at each trail location.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        .choices[0]
        .message.content
    )

    example = Example(
        input=prompt,
        actual_output=str(response),
        retrieval_context=[str(vector_db_context), str(research_data)],
    )
    judgment.async_evaluate(
        scorers=[FaithfulnessScorer(threshold=0.5)],
        example=example,
        model="gpt-4o",
    )

    judgement_client.run_evaluation(
        scorers=[ItineraryStructureScorer(threshold=0.5)],
        examples=[example],
        eval_run_name="ItineraryStructureScorer",
        # override=True,
        project_name="travel_agent_demo",
        model="gpt-4o",
    )

    return response


def clean_trail_data(
    trail_data: pd.DataFrame,
) -> str:
    """
    Clean the trail data.

    Args:
        trail_data (pd.DataFrame): The trail data to clean

    Returns:
        str: The cleaned trail data
    """
    trail_df = trail_data.copy()
    trail_df.drop(columns=["trail_id"], inplace=True)

    # Filter the trails that contain the word "[CLOSED]"
    trail_df = trail_df[
        ~trail_df["name"].str.contains(
            "[CLOSED]",
            regex=False,
        )
    ]

    trail_df["length"] = trail_df.apply(
        lambda x: str(round(x.length * 0.3048, 2)) + " m"
        if x.units == "i"
        else str(round(x.length, 2)) + " m",
        axis=1,
    )
    trail_df.rename(columns={"length": "trail_length"}, inplace=True)
    trail_df.drop(columns=["units"], inplace=True)

    # Convert the _geoloc column to a dictionary
    trail_df["_geoloc"] = trail_df["_geoloc"].apply(lambda x: ast.literal_eval(x))
    trail_df["lat"] = trail_df["_geoloc"].apply(lambda x: x["lat"])
    trail_df["lng"] = trail_df["_geoloc"].apply(lambda x: x["lng"])
    trail_df.drop(columns=["_geoloc"], inplace=True)

    trail_df["features"] = trail_df["features"].apply(lambda x: ast.literal_eval(x))
    trail_df["activities"] = trail_df["activities"].apply(lambda x: ast.literal_eval(x))

    if trail_df.empty:
        print("No trails found")
        exit(0)

    trail_json = convert_df_to_json(trail_df)

    return trail_json


def convert_df_to_json(df: pd.DataFrame) -> str:
    """
    Convert the trail data to a JSON string.

    Args:
        df (pd.DataFrame): The trail data to convert

    Returns:
        str: The JSON string
    """

    trail_list = []

    for _, row in df.iterrows():
        # Create a description dictionary with all columns except 'name'
        description = {
            "name": row["name"],
            "lat": row["lat"],
            "lng": row["lng"],
            "city_name": row["city_name"],
            "state_name": row["state_name"],
            "area_name": row["area_name"],
            "country_name": row["country_name"],
            "popularity": row["popularity"],
            "trail_length": row["trail_length"],
            "elevation_gain": row["elevation_gain"],
            "difficulty_rating": row["difficulty_rating"],
            "route_type": row["route_type"],
            "visitor_usage": row["visitor_usage"],
            "avg_rating": row["avg_rating"],
            "num_reviews": row["num_reviews"],
            "features": row["features"],
            "activities": row["activities"],
        }

        # Create the trail dictionary with only trail_name and description
        trail_dict = {
            "name": row["name"],
            "lat": row["lat"],
            "lng": row["lng"],
            "city_name": row["city_name"],
            "state_name": row["state_name"],
            "features": row["features"],
            "activities": row["activities"],
            "description": description,
        }
        trail_list.append(trail_dict)

    # Convert to JSON
    trail_json = json.dumps(trail_list, indent=2)
    return trail_json


@judgment.observe(span_type="function")
async def search_trail(
    source: str,
    trail_data: str,
    filter_input: List[str],
    filter_pref: str,
    difficulty: str,
) -> str:
    """
    Main function to generate a travel plan based on top 5 trails.

    Args:
        source (str): The source location
        trail_data (str): The trail data to populate the vector DB
        filter_input (List[str]): The activities and features to filter by
        filter_pref (str): The preference to filter by
        difficulty (str): The difficulty level to filter by

    Returns:
        str: The travel plan
    """
    research_data = await research_destination(
        source=source,
        trail_data=trail_data,
        filter_input=filter_input,
        filter_pref=filter_pref,
        difficulty=difficulty,
    )
    res = await create_travel_plan(
        source=source,
        research_data=research_data,
    )
    return res


if __name__ == "__main__":
    load_dotenv()

    # User input
    source = input("Enter your home location: ")
    activities_input = input("Enter your activities (comma-separated): ")
    features_input = input("Enter your features (comma-separated): ")
    filter_pref = input("Enter the filter preference (activities or features): ")
    difficulty = input("Enter the difficulty level (easy, moderate, hard): ")

    # source = "San Francisco"
    # filter_pref = "activities"
    # activities_input = "fishing"
    # features_input = "views, wildlife, dogs"
    # difficulty = "easy"

    # Parse comma-separated inputs into lists
    if filter_pref == "activities":
        filter_input = [activity.strip() for activity in activities_input.split(",")]
    else:
        filter_input = [feature.strip() for feature in features_input.split(",")]

    trail_data = pd.read_csv("./trails/custom_data/alltrails-data.csv")

    async def main():
        cleaned_trail_data = clean_trail_data(
            trail_data=trail_data,
        )

        travel_plan = await search_trail(
            source=source,
            trail_data=cleaned_trail_data,
            filter_input=filter_input,
            filter_pref=filter_pref,
            difficulty=difficulty,
        )
        print("\nGenerated Travel Plan:\n", travel_plan)

    asyncio.run(main())
