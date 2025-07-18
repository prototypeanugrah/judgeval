# Trail Agent Demo – Custom Scorers & Project Updates

This branch adds a complete trail-planning demo that showcases how to extend **JudgeVal** with domain-specific scorers.  
The demo lives in the `trails/` directory and highlights two new scorers:

| Scorer | Purpose | Located in |
|--------|---------|------------|
| **PreferenceMatchScorer** | Verifies that the recommended trails strictly match the user’s requested *activities* or *features*. | `trails/custom_scorers.py` |
| **ItineraryStructureScorer** | Grades the final travel plan for clear structure, inclusion of key trail data, and thoughtful use of weather info. | `trails/custom_scorers.py` |

## Key Code Changes

1. **New file** `trails/custom_scorers.py`  
   • Implements both scorers, subclassing JudgeVal’s `ExampleScorer` and `ClassifierScorer`.  
2. **Updated** `trails/trail_agent.py`  
   • Integrates the scorers into the research and itinerary generation pipeline.  
   • Adds ChromaDB retrieval, weather lookups, distance calculations, and async evaluation calls.  
3. **Data**  
   • Sample AllTrails CSV stored in `trails/custom_data/alltrails-data.csv` (cleaned at runtime).  
   • Persistent ChromaDB folder created under `trails/custom_data/chroma_db/`.

## Running the Demo

```bash
# Install Python deps
pip install -r requirements.txt   # or follow pyproject.toml

# Export required API keys
export OPENAI_API_KEY=...
export JUDGMENT_API_KEY=...
export TAVILY_API_KEY=...
export OPENWEATHER_API_KEY=...
export DISTANCEMATRIXFAST_API_KEY=...

# Launch the interactive script
python trails/trail_agent.py
```

The script will:
1. Clean and embed the trail dataset.
2. Ask the user for location, preferred activities/features, and difficulty.
3. Retrieve the top 5 matching trails, pull live weather & distance data, and generate an itinerary with GPT-4o.
4. Auto-evaluate the results with the custom scorers plus built-in JudgeVal scorers.

## Directory Overview

```
trails/
├── custom_data/            # CSV + persistent ChromaDB indices
├── custom_scorers.py       # PreferenceMatch & ItineraryStructure scorers
├── trail_agent.py          # End-to-end demo script
└── README.md               # (this file)
```

## Notes

* The demo focuses on showcasing **how to write and plug in custom scorers**, not production-grade routing logic.  
* Scorer thresholds are currently set to `0.5` for easy experimentation—tune as needed.  

Happy trail planning!

## Next steps (TODO)
- Implement a SummarizerScorer
- Experiment with Multi-modal scorers
