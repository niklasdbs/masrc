# Multiple Travelling Offivers Problem

## Dataset
Download the parking restriction and sensor data from https://data.melbourne.vic.gov.au/browse?tags=parking.
Then use create_dataset.py to preprocess these files into the correct format for the simulation.

## Simulation Environment
### Observations
Using the observation creators it is possible to derive various observations from the global state of the environment.

Note that some parts of the observation creators use cython. Build using 'python src/setup.py build_ext --inplace'

### Rendering
Rendering to a video requires ffmpeg.

## Available Agents
* Twin GRCN (OURS)
* Shared Independent using the single-agent SRC approach from cite
* COMA
* QMIX
* MARDAM
* LERK (based on todo)
* GREEDY
* Actor Critic
* Random Agent (useful for debugging/benchmark)

## Structure
* agents: Different agents
* trainers: various training modes

## Configuration
Everything is configured using hydra.

## How to run
python main.py -m +experiment=twin_grcn

#TODO moaf
#TODO remove all references to name (nst), strauss, dbs, ifi, lmu, ...