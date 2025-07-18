"""
Custom scorers for the trail agent.
"""

import json
from typing import List

from judgeval.constants import APIScorerType
from judgeval.data import Example, ExampleParams
from judgeval.scorers import ClassifierScorer
from judgeval.scorers.example_scorer import ExampleScorer


class PreferenceMatchScorer(ExampleScorer):
    """
    A scorer to verify that the recommended trails strictly match the user's
    requested activities and features.
    """

    name: str = "Preference Match Scorer"
    score_type: str = "preference_match"
    threshold: float = 0.5
    required_params: List[ExampleParams] = [
        ExampleParams.INPUT,
        ExampleParams.RETRIEVAL_CONTEXT,
    ]

    async def a_score_example(self, example: Example) -> float:
        """
        The scoring logic. This method should return the score as a float.
        """
        # We expect the user's preferences to be in the 'input' and the
        # retrieved metadata to be in the 'retrieval_context'.
        if not (
            example.input
            and isinstance(example.input, dict)
            and example.retrieval_context
        ):
            return 0.0

        required_activities = set(
            activity.lower() for activity in example.input.get("activities", [])
        )

        # The first item in retrieval_context will be our list of trail metadata
        recommended_trails = example.retrieval_context

        if not recommended_trails:
            return 0.0

        matches = 0
        for trail_metadata_str in recommended_trails:
            try:
                # Parse the JSON string into a dictionary
                trail_metadata = json.loads(trail_metadata_str)
                trail_activities = set(
                    activity.lower().strip()
                    for activity in trail_metadata.get("activities", "").split(",")
                )

                if required_activities.issubset(trail_activities):
                    matches += 1
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON: {trail_metadata_str}. Error: {e}")
                continue

        score = matches / len(recommended_trails)
        return score


class ItineraryStructureScorer(ClassifierScorer):
    """
    A classifier scorer to evaluate the quality and structure of the final itinerary.
    """

    def __init__(self, threshold: float = 0.5):
        prompt = """
        You are an expert travel guide who grades travel plans. The user was asked to create a structured travel itinerary.
        Based on the provided context (the original prompt given to the agent) and the actual output (the agent's generated itinerary),
        determine if the itinerary meets ALL of the following quality criteria:

        1.  **Clear Structure:** The plan is well-organized, using headings or bullet points for each recommended trail.
        2.  **Includes Key Data:** Each trail recommendation explicitly states its distance and estimated travel duration.
        3.  **Intelligent Weather Use:** The plan thoughtfully incorporates the provided weather data into the recommendations (e.g., "Since it might rain, start this trail early," or "This trail is a good choice for a sunny day.").

        If the itinerary meets ALL of these criteria, respond with 'Y'. Otherwise, respond with 'N' and provide a brief justification.

        **CONTEXT (PROMPT GIVEN TO AGENT):**
        {{input}}

        **ACTUAL OUTPUT (AGENT'S ITINERARY):**
        {{actual_output}}
        """

        super().__init__(
            name="Itinerary Structure Scorer",
            score_type=APIScorerType.PROMPT_SCORER,
            conversation=[{"role": "system", "content": prompt}],
            options={"Y": 1.0, "N": 0.0},
            threshold=threshold,
        )
