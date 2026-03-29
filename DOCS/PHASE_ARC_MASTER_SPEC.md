> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# ARC-AGI-3 Quickstart

> ARC-AGI-3 is an Interactive Reasoning Benchmark designed to measure an AI Agent's ability to generalize in novel, unseen environments.

<div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
  <div style={{ flex: 1 }}>
    <p>
      Traditionally, to measure AI, static benchmarks have been the yardstick.
      These work well for evaluating LLMs and AI reasoning systems. However, to evaluate frontier AI agent systems, we
      need new tools that measure:
    </p>

    <ul>
      <li>Exploration</li>
      <li>Percept → Plan → Action</li>
      <li>Memory</li>
      <li>Goal Acquisition</li>
      <li>Alignment</li>
    </ul>

    <p>
      By building agents that can play ARC-AGI-3, you're directly contributing
      to the frontier of AI research. <br /><br /> Learn more about{' '}
      <a href="https://arcprize.org/arc-agi/3">ARC-AGI-3</a>.
    </p>
  </div>

  <div style={{ flex: 1, textAlign: 'center' }}>
    <img src="https://mintcdn.com/arcprizefoundation/sx3SsV7kmM_q56IF/images/Ls20Human.gif?s=61025c7aeb245af080aba9e735a6f1cf" alt="Human playing LS20" width="512" height="512" data-path="images/Ls20Human.gif" />

    <p>
      Can you build an agent to beat{' '}
      <a href="https://arcprize.org/tasks/ls20">this game</a>?
    </p>
  </div>
</div>

## Play your first ARC-AGI-3 environment

### 1. Install the [ARC-AGI Toolkit](https://github.com/arcprize/arc-agi)

```bash  theme={null}
uv init
uv add arc-agi
# or
pip install arc-agi
```

### 2. Set your `ARC_API_KEY`

Optionally set your `ARC_API_KEY`. If no key is provided, an anonymous key will be used. However, registering for an API key will give you access to public games at release. [Get an ARC\_API\_KEY](/api-keys)

```bash  theme={null}
export ARC_API_KEY="your-api-key-here"
# or
echo 'ARC_API_KEY=your-api-key-here' > .env
```

### 3. Play your first game

Create a file called `my-play.py`:

```python  theme={null}
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

# Take a few actions
for _ in range(10):
    env.step(GameAction.ACTION1)

print(arc.get_scorecard())
```

Run it:

```bash  theme={null}
python play.py
```

You should see the game render in your terminal and a scorecard with your results.

🎉 Congratulations! You just played your first ARC-AGI-3 environment programatically.

Do you feel the AGI yet?

## Next Steps

After running your first environment:

1. **Make it fast** - Use `env = arc.make("ls20")` without `render_mode` to hit +2K FPS
2. **Try a different game** - Run `env = arc.make("ft09", render_mode="terminal")` to play a another game. See a list of games available at [arcprize.org/tasks](https://arcprize.org/tasks) or via the [ARC-AGI Toolkit](/toolkit/list-games)
3. **Use an agent** - Explore [agent templates](/llm_agents) or [create your own agent](/agents-quickstart).
4. **Explore the ARC-AGI Toolkit** - [The ARC-AGI Toolkit](./toolkit/overview) allows quick and easy integration with ARC-AGI Environments.


Built with [Mintlify](https://mintlify.com).

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# API Keys

> How to get and use your ARC-AGI-3 API key

## Why Get an API Key?

Registering for an API key allows you to:

* **Track your progress** across games and sessions
* **Access the full list of games** when launch goes out

## How to Get Your API Key

1. Go to [arcprize.org/platform](https://arcprize.org/platform)

2. Register by logging in with either **Google** or **GitHub**

3. Click on your **user profile** in the top right corner

4. In your user profile, find the **API Keys** section

5. Create a new key. This is your `ARC_AGI_API` key

Once you have your key, set it in your enviornment or `.env` and you'll have access to the entire set of public games once available on the platform.

## Using Your API Key

Set your API key as an environment variable:

```bash  theme={null}
export ARC_API_KEY="your-api-key-here"
```

Or add it to a `.env` file in your project:

```bash  theme={null}
echo 'ARC_API_KEY=your-api-key-here' > .env
```

The toolkit will automatically load your key from the environment when you create an `Arcade` instance:

```python  theme={null}
import arc_agi

# Automatically uses ARC_API_KEY from environment
arc = arc_agi.Arcade()

# Or pass the API key explicitly
arc = arc_agi.Arcade(arc_api_key="your-api-key-here")
```


Built with [Mintlify](https://mintlify.com).

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Game Schema

> Structure and format of ARC-AGI-3 game environments

ARC-AGI-3 games are turn-based environments where agents interact with 2D grids through a standardized action interface. Each game maintains state through discrete action-response cycles.

* Agents receive 1-N frames of JSON objects with the game state and metadata.
* Agents respond with an [action](/actions) to interact with the game.

## Grid Structure

* **Dimensions:** Maximum 64x64 grid size
* **Cell Values:** Integer values 0-15 representing different states/colors
* **Coordinate System:** (0,0) at top-left, (x,y) format

## Game ID Format

Game IDs are formatted as `<game_name>`-`<version>`.

`game_names` are stable, but `version` may change as games update.

## Game Available Actions

Each game provides an explicit set of available actions. The actions available vary per game and are stated explicitly so your agent knows what it can do.

To learn about the standardized action interface, see the [Actions](/actions) page.

To see how to retrieve a game's available actions programmatically, see [List Available Actions](/toolkit/list-actions).


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Available Games

> List of ARC-AGI-3 games and how to discover them

ARC-AGI-3 consists of a series of public games that are playable by both humans and AI agents.

## Discovering Games

To see a list of available games:

* Browse games at [arcprize.org/tasks](https://arcprize.org/tasks)
* Use the ARC-AGI Toolkit to [list games programmatically](/toolkit/list-games)

By default, three games are available to anonymous users after launch. An API key is <u>required</u> to access the remaining public games.

[Get a free API key](/api-keys) to unlock them.

## Example Games

* [ls20](https://arcprize.org/tasks/ls20) - Agent reasoning
* [ft09](https://arcprize.org/tasks/ft09) - Elementary Logic
* [vc33](https://arcprize.org/tasks/vc33) - Orchestration


Built with [Mintlify](https://mintlify.com).

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Actions

> Input interface for ARC-AGI-3 games

All games implement a standardized action interface with seven core actions:

| Action    | Description                                                                                   |
| --------- | --------------------------------------------------------------------------------------------- |
| `RESET`   | Initialize or restarts the game/level state                                                   |
| `ACTION1` | Simple action - varies by game (semantically mapped to up)                                    |
| `ACTION2` | Simple action - varies by game (semantically mapped to down)                                  |
| `ACTION3` | Simple action - varies by game (semantically mapped to left)                                  |
| `ACTION4` | Simple action - varies by game (semantically mapped to right)                                 |
| `ACTION5` | Simple action - varies by game (e.g., interact, select, rotate, attach/detach, execute, etc.) |
| `ACTION6` | Complex action requiring x,y coordinates (0-63 range)                                         |
| `ACTION7` | Simple action - Undo (e.g., interact, select)                                                 |

### Human Player Keybindings

When playing games manually in the ARC-AGI-3 UI, you can use these keyboard shortcuts instead of clicking action buttons:

| Control Scheme     | ACTION1 | ACTION2 | ACTION3 | ACTION4 | ACTION5 | ACTION6     | ACTION7    |
| ------------------ | ------- | ------- | ------- | ------- | ------- | ----------- | ---------- |
| **WASD + Space**   | `W`     | `S`     | `A`     | `D`     | `Space` | Mouse Click | CTRL/CMD+Z |
| **Arrow Keys + F** | `↑`     | `↓`     | `←`     | `→`     | `F`     | Mouse Click | CTRL/CMD+Z |

All control schemes support mouse clicking for ACTION6 (coordinate-based actions). Choose whichever scheme feels most comfortable for your playstyle.

### Available Actions

Each game explicitly defines the set of available actions that can be used within that game. This approach ensures clarity for both human and AI participants by making it clear which actions are permitted, thereby reducing confusion. In the human-facing UI, available actions are visually highlighted or dismissed to provide the same affordance.

For each action taken, the metadata of the returned frame will indicate which actions are available. Agents may use this information to narrow the action space and develop effective strategies for completing the game.

Note: Action 6 does not provide explicit X/Y coordinates for active areas. If Action 6 is available, only its availability will be indicated, without specifying which coordinates are active.


Built with [Mintlify](https://mintlify.com).

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# ARC-AGI-3 Scoring Methodology

> How ARC-AGI-3 scoring works

ARC-AGI-3 uses **Relative Human Action Efficiency** (RHAE, pronounced "ray") to score AI systems.

RHAE measures per-level action efficiency compared to a human baseline, normalized per game, across all games.

## What Gets Measured

AI is scored on two criteria:

1. **Completion** — How many levels did the AI complete in each game?
2. **Efficiency** — How many actions did the AI take compared to humans?

## What Counts as an Action

An *action* is a discrete interaction with the environment. Each turn where the agent submits a command, move, or input that affects the game state counts as an action.

Internal operations that do not alter the environment (tool calls, reasoning steps, retries) are **not counted** as actions.

## Human Baseline

Human baselines are established through controlled testing where participants play each ARC-AGI-3 game for the first time (having never seen the game before). For each game, multiple first-time players are observed, and the **2nd best human** (fewest actions) per game is recorded as the baseline.

Using the 2nd best human:

* Removes outlier winners while still representing proficient human performance
* Avoids penalizing for early misclicks
* Keeps the baseline grounded in real play, not theoretical speed-runs

## How Scoring Works

### Per-Level Scoring

For each level the AI completes, calculate:

```
level_score = (human_baseline_actions / ai_actions) ^ 2
```

* If human baseline is 10 actions and AI takes 10 → level score is 1.0 (100%)
* If human baseline is 10 actions and AI takes 20 → level score is 0.25 (50%)
* If human baseline is 10 actions and AI takes 1,00 → level score is 0.01 (1%)

### Per-Level Score Cap

The maximum score per level is capped at **1.0x** human baseline. If an AI discovers a shortcut and completes a level faster than humans, it still only receives 1.0.

This encourages building AI that **generalizes across games** rather than exploiting individual levels.

### Per-Game Aggregation

The game score is the **weighted average** of all per-level scores, using a 1 index of the level as the weight, for that game.

**Example:** A game has 7 levels. The AI scores:

* Levels 1-3: 0.25 each (took twice as many actions as human)
* Levels 4-7: 0 each (did not complete the level)

Game score = (0.25x1 + 0.25x2 + 0.5x3 + 0x4 + 0x5 + 0x6 + 0x7) / (1+2+3+4+5+6+7) = **0.01289 (1.29%)**

Per-level weighted average underweights the starting levels which are tutorial/easy levels and overweights the more difficult levels where mastery must be demonstrated.

### Total Score

Total score is the **average of all game scores**, resulting in a final score between 0% and 100%.

## Score Interpretation

| Score | Interpretation                                                                |
| ----- | ----------------------------------------------------------------------------- |
| 100%  | AI completes all games/levels while matching or surpassing human efficiency   |
| 1-99% | A mixture of level completion rates and efficiency relative to human baseline |
| 0%    | AI never completes a level across any game                                    |


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Scorecards

> Keeping track of agent performance

Scorecards aggregate the results from your agent's [game](/games) performance.

## Ways to Get a Scorecard

* **Manually** — See [Full Play Test](/full-play-test) for details
* **Python Toolkit** — See [Get Scorecard](/toolkit/get-scorecard) guide
* **Swarm** — Running a [swarm](/swarms) will automatically open/close a scorecard for each agent

For game runs done via the API, scorecards can be viewed online at [https://arcprize.org/scorecards](https://arcprize.org/scorecards) and [https://arcprize.org/scorecards/\`scorecard\_id\`](https://arcprize.org/scorecards/`scorecard_id`).

## Scorecard Fields

| Field       | Description                                                                                        |
| ----------- | -------------------------------------------------------------------------------------------------- |
| tags        | Array of strings used to categorize and filter scorecards (e.g., \["experiment1", "v2.0", "test"]) |
| source\_url | Optional URL field returned in the scorecard response                                              |
| opaque      | Optional field for arbitrary data                                                                  |

```python  theme={null}
import arc_agi

arc = arc_agi.Arcade()

scorecard_id = arc.create_scorecard(
    tags=["experiment", "my-awesome-agent-v5-final-final"],
    source_url="https://github.com/my/repo",
    opaque={"custom_field": "any data"}
)
```

For more information, see [Create Scorecard](/toolkit/create-scorecard).

## Sharing

Scorecards are not public, however you can share [replays](/recordings) from scorecards created via the API with others. Local scorecards cannot be shared.

### Notes

* Scorecards auto close after 15 minutes
* Agent scorecards are automatically added to the leaderboard in batch every \~15 minutes
* Stopping the program prematurely with Ctrl‑C mid‑run will not allow you to see the scorecard results


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Recordings & Replays

> Viewing your agent's gameplay

Recordings let you view and share your agent's gameplay sessions.

## When Recordings Are Available

| Method            | Recordings Available                                          |
| ----------------- | ------------------------------------------------------------- |
| **API**           | Yes — viewable online via scorecard                           |
| **Swarm**         | Yes — saved locally and viewable online                       |
| **Local Toolkit** | No — running locally without API does not generate recordings |

## Online Replays

For games played via the API, you can view recordings online through your scorecard:

`https://arcprize.org/scorecards/<scorecard_id>`

Here is an example [recording](https://arcprize.org/replay/1d251d20-9043-4ace-9f9d-09822f5438d8).

## Local Recording Files

When running a [swarm](/swarms), agent gameplay is recorded by default and stored in the `recordings/` directory with GUID-based filenames:

```
ls20-6cbb1acf0530.random.100.a1b2c3d4-e5f6-7890-abcd-ef1234567890.recording.jsonl
```

The filename format is: `{game_id}.{agent_type}.{max_actions}.{guid}.recording.jsonl`

## Recording File Format

### JSONL Format

Recordings are stored in JSONL format with timestamped entries:

```json  theme={null}
{"timestamp": "2024-01-15T10:30:45.123456+00:00", "data": {"game_id": "ls20-016295f7601e", "frame": [...], "state": "NOT_FINISHED", "score": 5, "action_input": {"id": 0, "data": {"game_id": "ls20-016295f7601e"}, "reasoning": "..."}, "guid": "...", "full_reset": false}}
{"timestamp": "2024-01-15T10:30:46.234567+00:00", "data": {"game_id": "ls20-016295f7601e", "frame": [...], "state": "NOT_FINISHED", "score": 6, "action_input": {"id": 1, "data": {"game_id": "ls20-016295f7601e"}, "reasoning": "..."}, "guid": "...", "full_reset": false}}
```


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Agents Quickstart

> Build AI agents for ARC-AGI-3 environments

Get started with the [ARC-AGI-3 Agents repo](https://github.com/arcprize/ARC-AGI-3-Agents).

Watch the [Agent Quickstart tutorial video](https://www.youtube.com/watch?v=xEVg9dcJMkw).

## Step 1: Clone the Repo

```bash  theme={null}
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

Make sure you have your `ARC_API_KEY` populated in your environment variables. See [API Keys](/api-keys) for setup instructions.

## Step 2: Run an Agent

Run the random agent on the `ls20` game:

```bash  theme={null}
uv run main.py --agent=random --game=ls20
```

The [replay](/recordings) of your agent is available at the end of the run.

## Next Steps

* [Create Your Own Agent](/create-agent) — Build a custom agent
* [LLM Agents](/llm_agents) — Build agents powered by LLMs
* [Partner Templates](/partner_templates/agentops) — Start from partner-provided templates


Built with [Mintlify](https://mintlify.com).



> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Benchmarking Tooling (BETA)

> Run repeatable agent evaluations.

Currently in beta, our Benchmarking Agent will be the standard way to measure AI performance across model providers.

## ARC Harness `arcagi3`

This is a developer harness for building and benchmarking agentic research workflows on the **ARC-AGI-3** corpus of environments.

## When to use it

* Compare model versions or prompt strategies on the same game set.
* Detect regressions after code or prompt changes.
* Generate official scorecards and replays for sharing.
* Experimenting with multiple custom agentic architectures.

## Quickstart

### Prerequisites

* **Python**: `3.9+`
* **uv**: recommended package manager. Install from [uv.pm](https://github.com/astral-sh/uv) or `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **ARC-AGI-3 API key**: required to talk to the ARC server. Sign up for a key [here](https://arcprize.org/platform)

### Install

Clone the repository:

```bash  theme={null}
git clone git@github.com:arcprize/arc-agi-3-benchmarking.git
cd arc-agi-3-benchmarking
```

From repo root:

```bash  theme={null}
uv venv
uv sync
```

This creates a virtual environment (if needed) and installs the project and dependencies in editable mode.

Alternatively, without `uv`:

```bash  theme={null}
pip install -e .
```

### Setting up your environment

Set the ARC API key and your provider keys. You can put them in a `.env` file (see [`.env.example`](https://github.com/arcprize/arc-agi-3-benchmarking/blob/main/.env.example)) or export them in your shell.

Provider key links:

* [OpenAI](https://platform.openai.com/account/api-keys)
* [Anthropic](https://console.anthropic.com/account/api-keys)
* [Google Gemini](https://console.cloud.google.com/apis/credentials)
* [OpenRouter](https://openrouter.ai/api-keys)
* [Fireworks](https://app.fireworks.ai/account/api-keys)
* [Groq](https://groq.com/account/api-keys)
* [DeepSeek](https://console.deepseek.com/account/api-keys)
* [Hugging Face](https://huggingface.co/settings/tokens)

Check configuration:

```bash  theme={null}
uv run python -m arcagi3.runner --check
```

### Select your game

```bash  theme={null}
uv run python -m arcagi3.runner --list-games
```

### Pick your model

```bash  theme={null}
uv run python -m arcagi3.runner --list-models
```

### Benchmark

```bash  theme={null}
uv run python -m arcagi3.runner \
  --game_id ls20 \
  --config gpt-5-2-openrouter \
  --max_actions 3
```

### Scorecards

When you run a benchmark, a scorecard is saved on the ARC server. If you are logged in, you can view them at [arcprize.org/scorecards](https://arcprize.org/scorecards).

## Learn More

The [benchmarking README](https://github.com/arcprize/arc-agi-3-benchmarking#readme) has more information than what is published here. Be sure to also reference how to [create your own agent](https://github.com/arcprize/arc-agi-3-benchmarking/blob/main/docs/create_agent.md) to start experimenting with new agentic architectures.


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# HuggingFace Template

> HuggingFace Template

These templates provide a great starting point for building agents with the [HuggingFace](https://huggingface.co/) ecosystem. Check out the [smolagents documentation](https://huggingface.co/docs/smolagents/index) to learn more.

* **`SmolCodingAgent`**: A text-based agent that uses a code-generating model to reason about the game and execute actions as Python code.
* **`SmolVisionAgent`**: A multimodal agent that processes game frames as images, allowing it to "see" the game state.

To run these agents, use the following commands:

```bash  theme={null}
# Run the text-based coding agent
uv run main.py --agent=smolcodingagent --game=ls20

# Run the vision-based agent
uv run main.py --agent=smolvisionagent --game=ls20
```

[Source File](https://github.com/arcprize/ARC-AGI-3-Agents/blob/main/agents/templates/smolagents.py)


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Local vs Online

> Playing games locally with the engine vs online via the API

Choose how you want to run ARC-AGI-3 games.

<CardGroup cols={2}>
  <Card title="Local (Recommended)" icon="computer" href="#local">
    Fast, no rate limits, run many instances
  </Card>

  <Card title="Online" icon="cloud" href="#online">
    Scorecards, replays, requires API key
  </Card>
</CardGroup>

## Local

Run games locally using the ARC-AGI engine. This is the recommended approach for development and testing.

```python  theme={null}
from arc_agi import Arcade, OperationMode

arc = Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("ls20", render_mode="terminal")
```

| Advantages                              | Limitations          |
| --------------------------------------- | -------------------- |
| \~2,000 FPS (120,000 frames per minute) | No online scorecards |
| No rate limits                          | No shareable replays |
| Run as many instances as you want       |                      |
| No API key required                     |                      |

## Online

Run games via the API to get scorecards and replays.

```python  theme={null}
from arc_agi import Arcade, OperationMode

arc = Arcade(operation_mode=OperationMode.ONLINE)
env = arc.make("ls20", render_mode="terminal")
```

| Advantages                            | Limitations                                       |
| ------------------------------------- | ------------------------------------------------- |
| View [scorecards](/scorecards) online | Requires [API key](/api-keys)                     |
| Shareable [replays](/recordings)      | Capped at [600 requests per minute](/rate_limits) |
| Results appear on leaderboard         |                                                   |

## Learn More

For all operation mode options and configuration details, see [operation\_mode](/toolkit/arc_agi#operation_mode) in the Toolkit reference.


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Full Play Test

> What happens under the hood when running an ARC-AGI-3 game

This page demonstrates how to interact with the ARC-AGI-3 API directly using HTTP requests. This is useful for understanding what happens "under the hood" when running a game.

<Note>
  **Recommended approach:** For most use cases, we recommend using the [ARC-AGI Toolkit](/toolkit/minimal) instead of calling the API directly. The Toolkit handles authentication, error handling, and provides a simpler interface.
</Note>

## Direct API Workflow

Below is the full workflow for running a game programmatically via the API.

## Game State Enumeration

| State          | Description                                                        |
| -------------- | ------------------------------------------------------------------ |
| `NOT_FINISHED` | Game is active and awaiting next action                            |
| `WIN`          | Objective completed successfully                                   |
| `GAME_OVER`    | Game terminated due to the max actions reached or other conditions |

## Full Playtest Example

This is a bare-bones example (for educational purposes) also available as a [notebook](https://colab.research.google.com/drive/1Bt4PU6Xl_avLPV70hNAyReXaRqFDhifJ?usp=sharing).

```python  theme={null}
#!/usr/bin/env python3
"""
Simple demo showing what a swarm agent does under the hood.
This is a bare-bones example for educational purposes.
"""

import json
import os
import random
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

# Setup
ROOT_URL = "https://three.arcprize.org"
API_KEY = os.getenv("ARC_API_KEY")

# Create a session with headers
session = requests.Session()
session.headers.update({
    "X-API-Key": API_KEY,
    "Accept": "application/json"
})

print("=== MANUAL SWARM DEMO ===")
print("This shows what happens when an agent plays an ARC game.\n")

# Step 1: Get available games
print("STEP 1: Getting list of games...")
response = session.get(f"{ROOT_URL}/api/games")
games = [g["game_id"] for g in response.json()]
print(f"Found {len(games)} games")

# Pick a random game
game_id = random.choice(games)
print(f"Selected game: {game_id}\n")

# Step 2: Open a scorecard (tracks performance)
print("STEP 2: Opening scorecard...")
response = session.post(
    f"{ROOT_URL}/api/scorecard/open",
    json={"tags": ["manual_demo"]}
)
card_id = response.json()["card_id"]
print(f"Scorecard ID: {card_id}\n")

# Step 3: Start the game
print("STEP 3: Starting game with RESET action...")
url = f"{ROOT_URL}/api/cmd/RESET"
print(f"URL: {url}")
response = session.post(
    url,
    json={
        "game_id": game_id,
        "card_id": card_id
    }
)

# Check if response is valid
if response.status_code != 200:
    print(f"Error: {response.status_code} - {response.text}")
    exit()

game_data = response.json()
guid = game_data["guid"]
state = game_data["state"]
score = game_data.get("score", 0)
print(f"Game started! State: {state}, Score: {score}\n")

# Step 4: Play with random actions (max 5 actions)
print("STEP 4: Taking random actions...")
actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]

for i in range(5):
    # Check if game is over
    if state in ["WIN", "GAME_OVER"]:
        print(f"\nGame ended! Final state: {state}, Score: {score}")
        break

    # Pick a random action
    action = random.choice(actions)

    # Build request data
    request_data = {
        "game_id": game_id,
        "card_id": card_id,
        "guid": guid
    }

    # ACTION6 needs x,y coordinates
    if action == "ACTION6":
        request_data["x"] = random.randint(0, 29)
        request_data["y"] = random.randint(0, 29)
        print(f"Action {i+1}: {action} at ({request_data['x']}, {request_data['y']})", end="")
    else:
        print(f"Action {i+1}: {action}", end="")

    # Take the action
    response = session.post(
        f"{ROOT_URL}/api/cmd/{action}",
        json=request_data
    )

    game_data = response.json()
    state = game_data["state"]
    score = game_data.get("score", 0)
    print(f" -> State: {state}, Score: {score}")

# Step 5: Close scorecard
print("\nSTEP 5: Closing scorecard...")
response = session.post(
    f"{ROOT_URL}/api/scorecard/close",
    json={"card_id": card_id}
)
scorecard = response.json()
print("Scorecard closed!")
print(f"\nView results at: {ROOT_URL}/scorecards/{card_id}")

print("\n=== DEMO COMPLETE ===")
print("\nThis is what every agent does:")
print("1. Get games list")
print("2. Open a scorecard")
print("3. Reset to start the game")
print("4. Take actions based on its strategy (we used random)")
print("5. Close the scorecard when done")
print("\nThe real agents use smarter strategies instead of random!")
```

This workflow ensures your plays are tracked officially. For parallel playtests across games, use a [swarm](/swarms) to handle the orchestration automatically.


Built with [Mintlify](https://mintlify.com).

THE NOTEBOOK FOR THIS TEST IS IN THE NOTEBOOK FOLDER IN FLUX DIRECTOY notebooks/ARC_AGI_3_Agent_Manual_Playthrough.ipynb

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# ARC-AGI Toolkit Quickstart

> Getting started with the ARC-AGI Toolkit

The ARC-AGI Toolkit is an open-source Python SDK for ARC-AGI-3 environments, geared towards researchers looking to make progress on ARC-AGI-3.

The Toolkit enables:

* **Local development** — Run your agents locally without needing the API, built on top of the [ARC-AGI-3 game engine](https://github.com/arcprize/ARCEngine)
* **Customization** — Edit existing games and create new ones
* **Flexibility** — Interact with environments locally or via API

## QuickStart

### 1. Install the Toolkit

```bash  theme={null}
uv add arc-agi
# or
pip install arc-agi
```

### 2. Set your API key (optional)

```bash  theme={null}
export ARC_API_KEY="your-api-key-here"
```

If no key is provided, an anonymous key will be used. See [API Keys](/api-keys) for more details.

### 3. Play a game

```python  theme={null}
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

# See available actions
print(env.action_space)

# Take an action
obs = env.step(GameAction.ACTION1)

# Check your scorecard
print(arc.get_scorecard())
```

## Next Steps

* [Minimal Example](./minimal) — A complete script example
* [List Games](./list-games) — Discover available games
* [Local vs Online](/local-vs-online) — Choose how to run games

## Changelog

## \[0.9.3] - 2026-03-09

### Added

* `OperationMode.COMPETITION` method, see [Documentation](./competition_mode)
* Official Scoring
  * Average for an individual games is now weighted by the level index (1 indxed)
  * Score for an individual level is now squared.  A score of `0.5` now becomes `0.25`

### Fixed

* Continued fixes for 404 Scorecard not found

## \[0.9.2] - 2026-02-26

### Added

* `listen_and_serve` method, see [Documentation](./listen_and_serve)

### Fixed

* 404 Scorecard not found about 50% of the time when in `ONLINE` mode
* Game source being downloaded even if local copy already exists

## \[0.9.1] - 2026-01-29

Initial Release


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Sample ARC-AGI Toolkit Agent

> An example agent that interacts with ARC-AGI-3 locally

Here's a minimal example that plays a game and renders it in the terminal:

```python  theme={null}
import random

from arcengine import GameAction, GameState
import arc_agi

# Initialize the ARC-AGI-3 client
arc = arc_agi.Arcade()

# Create an environment with terminal rendering
env = arc.make("ls20", render_mode="terminal")
if env is None:
    print("Failed to create environment")
    exit(1)

# Play the game
for step in range(100):
    # Choose a random action
    action = random.choice(env.action_space)
    action_data = {}
    if action.is_complex():
        action_data = {
            "x": random.randint(0, 63),
            "y": random.randint(0, 63),
        }        
        
    # Perform the action (rendering happens automatically)
    obs = env.step(action, data=action_data)
    
    # Check game state
    if obs and obs.state == GameState.WIN:
        print(f"Game won at step {step}!")
        break
    elif obs and obs.state == GameState.GAME_OVER:
        env.reset()

# Get and display scorecard
scorecard = arc.get_scorecard()
if scorecard:
    print(f"Final Score: {scorecard.score}")
```

<Note>
  This example uses `render_mode="terminal"` to display the game in your terminal. If the game appears wrapped or distorted, try enlarging your terminal window or zooming out (Cmd/Ctrl + minus). For other rendering options, see [Render Games](./render-games).
</Note>


Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Competition Mode

> Running in Competition Mode

## Overview

This mode is **REQUIRED** to show up on the Unverified leaderboard and forces the following behavior.

* Environments must be interacted with via the API
* Scoring is against all available environments, even if you choose not to interact with them
* Only *Level Resets* are premitted, *Game Resets* are not allowed and become *Level Resets*
* Can only interact (call `make`) a single time for each environment
* Can only open a single Scorecard
* Cannot get scoring of an inflight scorecard, `get_scorecard` does not work

**Note:** The Kaggle Compeition is forced into this mode.

### Example

```
from arc_agi import Arcade, OperationMode

arc = Arcade(operation_mode=OperationMode.COMPETITION)
```

You can also set the ENVVAR of `OPERATION_MODE=COMPETITION`.


Built with [Mintlify](https://mintlify.com).



> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Listen And Serve

> Running the REST Endpoints to interact with Environments.

## Overview

Start a blocking Flask server that exposes the REST API. Uses `arc_agi.server.create_app()` under the hood.  This conforms to the [Rest API](https://docs.arcprize.org/rest_overview) to allow local execution for interactions with languages other than Python or with this Toolkit running in `ONLINE` mode.

**Parameters:**

* `host` (`str`, optional): Bind address. Default `"0.0.0.0"` to accept connections from any interface.
* `port` (`int`, optional): Port to listen on. Default `8001`.
* `competition_mode` (`bool`, optional): If `True`, enable competition mode. Default `False`.
* `save_all_recordings` (`bool`, optional): If `True`, save recordings for all runs. Default `False`.
* `add_cookie` (`Callable[[Response, str], Response]`, optional): Callback to inject a cookie into API responses. Receives `(response, api_key)`; must return the modified response. Use for session stickiness (e.g. ALB app cookies).
* `scorecard_timeout` (`int`, optional): Idle timeout in seconds before scorecards are auto-closed. If set, starts a background cleanup loop.
* `on_scorecard_close` (`Callable[[EnvironmentScorecard], None]`, optional): Callback invoked when a scorecard is closed (manually or by timeout).
* `extra_api_routes` (`Callable[[Arcade, Flask], None]`, optional): Callback to register custom routes. Receives `(arcade, app)`.
* `renderer` (`Callable[[int, FrameDataRaw], None]`, optional): Callback invoked for each frame during gameplay. Receives `(step_index, frame_data)`. Use for logging, visualization, or custom display.
* `**kwargs`: Passed through to `Flask.run()` (e.g. `debug=True`, `threaded=True`).

**Example (basic):**

```python  theme={null}
arc = Arcade()
arc.listen_and_serve(port=8001)
```

**Example (with `add_cookie` for session stickiness):**

```python  theme={null}
from flask import Response

def add_session_cookie(resp: Response, api_key: str) -> Response:
    resp.set_cookie("APPLICATION_COOKIE", api_key, path="/", httponly=True)
    return resp

arc.listen_and_serve(add_cookie=add_session_cookie)
```

**Example (with `on_scorecard_close`):**

```python  theme={null}
def on_close(scorecard):
    print(f"Scorecard closed: {scorecard.score}")

arc.listen_and_serve(on_scorecard_close=on_close)
```

**Example (with `extra_api_routes`):**

```python  theme={null}
def register_custom(arcade, app):
    @app.route("/custom")
    def custom():
        return {"environments": len(arcade.available_environments)}

arc.listen_and_serve(extra_api_routes=register_custom)
```

**Example (with `renderer` for logging):**

```python  theme={null}
def log_frame(step: int, frame_data):
    print(f"Step {step}: state={frame_data.state}, levels_completed={frame_data.levels_completed}")

arc.listen_and_serve(renderer=log_frame)
```


Built with [Mintlify](https://mintlify.com).



> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Rate Limits

> Limiting RPM for your agents

The ARC-AGI API is currently free to use during its research preview and is supported on a best-effort basis. We do not currently offer a formal SLA, uptime guarantee, or guaranteed response times. To prevent abuse and ensure fair access, we implement rate limits with an exponential backoff mechanism.

These limits help maintain system stability by throttling excessive requests. If you encounter rate limiting, your requests will receive a backoff response, requiring you to wait increasingly longer periods before retrying.

Rate limits are set at 600 requests per minute (RPM).

## Requesting Limit Increases

We are open to discussing increases in rate limits, particularly for researchers and teams requiring higher throughput.

If you need elevated limits, please email us at [team@arcprize.org](mailto:team@arcprize.org) with the subject line "Increase Rate Limits" to initiate a conversation.

## Navigating Rate Limits

If you've managed to hit a rate limit, you'll see a standard `429` response.

```json  theme={null}
{"error":"RATE_LIMIT_EXCEEDED","message":"rate limit has been exceeded"}
```


Built with [Mintlify](https://mintlify.com).

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# List available games

> Retrieves metadata for every game currently exposed by the
ARC-AGI-3 platform.  
Use this discovery endpoint to obtain `game_id` values before
opening a scorecard or issuing commands. Results are returned as
a flat array ordered alphabetically by `title`.




## OpenAPI

````yaml /arc3v1.yaml get /api/games
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/games:
    get:
      tags:
        - Games
      summary: List available games
      description: |
        Retrieves metadata for every game currently exposed by the
        ARC-AGI-3 platform.  
        Use this discovery endpoint to obtain `game_id` values before
        opening a scorecard or issuing commands. Results are returned as
        a flat array ordered alphabetically by `title`.
      operationId: listGames
      responses:
        '200':
          description: Successful lookup; array of game descriptors.
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Game'
              examples:
                sample:
                  summary: Two games
                  value:
                    - game_id: ls20-016295f7601e
                      title: LS20
                    - game_id: ft09-16726c5b26ff
                      title: FT09
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    Game:
      type: object
      description: >
        Human-readable name/identifier pair for an ARC-AGI-3 game.  

        Used when listing available titles or embedding game metadata in other
        payloads.
      properties:
        game_id:
          type: string
          example: ls20-016295f7601e
          description: Stable, globally unique ID combining slug and version/hash.
        title:
          type: string
          example: LS20
          description: Display title shown in UIs and scorecards.
      required:
        - game_id
        - title
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Open scorecard

> Creates a new scorecard to aggregate statistics across one or more
plays. The server returns a `card_id`, which must be included in all
subsequent RESET commands and in the final **/scorecard/close** call.
You may attach optional metadata (URL, tags, opaque JSON) that will
be echoed back in summary responses.




## OpenAPI

````yaml /arc3v1.yaml post /api/scorecard/open
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/scorecard/open:
    post:
      tags:
        - Scorecards
      summary: Open scorecard
      description: |
        Creates a new scorecard to aggregate statistics across one or more
        plays. The server returns a `card_id`, which must be included in all
        subsequent RESET commands and in the final **/scorecard/close** call.
        You may attach optional metadata (URL, tags, opaque JSON) that will
        be echoed back in summary responses.
      operationId: openScorecard
      requestBody:
        description: Optional metadata to associate with the new scorecard.
        required: false
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/OpenScorecardRequest'
            examples:
              minimal:
                summary: Minimal request
                value: {}
              full:
                summary: With tags, link, and opaque blob
                value:
                  source_url: https://github.com/example/agent
                  tags:
                    - baseline
                    - gpt-4o
                  opaque:
                    model: gpt-4o
                    temperature: 0.25
      responses:
        '200':
          description: scorecard successfully created.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/OpenScorecardResponse'
              examples:
                success:
                  value:
                    card_id: 8bb3b1b8-4b46-4a29-a13b-ad7850a0f916
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    OpenScorecardRequest:
      type: object
      description: |
        Optional metadata sent when opening a scorecard.  
        Every field is optional; omit any you don't need.  
        Use this to attach provenance links, free-form tags, or an
        “opaque” JSON blob describing the run (e.g. model settings,
        hyper-parameters, experiment notes). The opaque payload must not
        exceed 16 KB once serialized.
      properties:
        source_url:
          type: string
          format: uri
          description: Link to code, notebook, or write-up associated with the run.
        tags:
          type: array
          description: Arbitrary labels for later filtering and aggregation.
          items:
            type: string
        opaque:
          type: object
          description: |
            Free-form JSON data (≤ 16 KB). Stored verbatim; the service
            does not inspect or validate its structure.
          additionalProperties: true
    OpenScorecardResponse:
      type: object
      description: |
        Response returned after a successful “open scorecard” request.
        Contains the server-generated identifier for this tracked run.
      properties:
        card_id:
          type: string
          description: Globally unique ID for the newly opened scorecard.
      required:
        - card_id
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Close scorecard

> Finalises a previously opened scorecard, locking its data and
returning the aggregate results.  
After a scorecard is closed, additional RESET or ACTION commands
using its `card_id` are rejected.  
You must supply the `card_id` obtained from **/scorecard/open**.




## OpenAPI

````yaml /arc3v1.yaml post /api/scorecard/close
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/scorecard/close:
    post:
      tags:
        - Scorecards
      summary: Close scorecard
      description: |
        Finalises a previously opened scorecard, locking its data and
        returning the aggregate results.  
        After a scorecard is closed, additional RESET or ACTION commands
        using its `card_id` are rejected.  
        You must supply the `card_id` obtained from **/scorecard/open**.
      operationId: closeScorecard
      requestBody:
        description: Identifier of the scorecard to close.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CloseScorecardRequest'
            examples:
              example:
                value:
                  card_id: 8bb3b1b8-4b46-4a29-a13b-ad7850a0f916
      responses:
        '200':
          description: scorecard closed; final aggregate results returned.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ScorecardSummary'
              examples:
                success:
                  value:
                    card_id: 8ae52f21-b40c-457b-9f4e-65bd8381e67f
                    score: 100
                    source_url: https://sandbox.internal.arc-prize.com
                    tags:
                      - human
                    user_name: distracted_poincare
                    user_id: 102214435702678430912@google
                    published_at: '2026-01-26T23:34:57.097896Z'
                    environments:
                      - id: am92-80effacb
                        runs:
                          - id: am92-80effacb
                            guid: 4a38a278-796b-4f42-a28b-a27a68dbf862
                            score: 100
                            levels_completed: 5
                            actions: 136
                            resets: 0
                            state: WIN
                            completed: true
                            level_scores:
                              - 100
                              - 100
                              - 100
                              - 100
                              - 100
                            level_actions:
                              - 12
                              - 34
                              - 41
                              - 26
                              - 23
                            level_baseline_actions:
                              - 20
                              - 40
                              - 50
                              - 55
                              - 60
                            number_of_levels: 0
                            number_of_environments: 0
                        score: 100
                        actions: 136
                        levels_completed: 5
                        completed: true
                        level_count: 5
                        resets: 0
                    tags_scores:
                      - id: change9
                        guid: d3a0a4d8-536c-479d-b5a8-10bf1dc8aee3
                        score: 100
                        levels_completed: 5
                        actions: 136
                        resets: 0
                        state: NOT_FINISHED
                        completed: false
                        number_of_levels: 5
                        number_of_environments: 1
                      - id: example
                        guid: 66fe6b58-adf8-45b1-ac73-70ed486e22d0
                        score: 100
                        levels_completed: 5
                        actions: 136
                        resets: 0
                        state: NOT_FINISHED
                        completed: false
                        number_of_levels: 5
                        number_of_environments: 1
                    open_at: '2026-01-26T23:34:18.213873Z'
                    last_update: '2026-01-26T23:34:53.229182Z'
                    total_environments_completed: 1
                    total_environments: 1
                    total_levels_completed: 5
                    total_levels: 5
                    total_actions: 136
        '401':
          description: Missing or invalid **X-API-Key** header.
        '404':
          description: Supplied `card_id` does not correspond to an open scorecard.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    CloseScorecardRequest:
      type: object
      description: |
        Payload for closing a previously opened scorecard and finalising
        its aggregated results.
      properties:
        card_id:
          type: string
          description: |
            Identifier of the scorecard to close—use the `card_id`
            returned by **OpenScorecardResponse**.
      required:
        - card_id
    ScorecardSummary:
      type: object
      description: >
        Aggregate results for an entire scorecard run.  

        Returned when closing a scorecard or when retrieving a scorecard (open
        or closed).

        Includes cumulative totals, optional metadata echoed from the open
        request

        (e.g. `source_url`, `tags`, `opaque`), user identity (`user_name`,
        `user_id`),

        timestamps (`open_at`, `last_update`, `published_at`), and a
        per-environment

        breakdown in `environments`. The `tags_scores` array provides per-tag
        aggregates

        for runs that were tagged.
      properties:
        card_id:
          type: string
          description: The scorecard ID returned by **OpenScorecardResponse**.
        score:
          type: integer
          description: Aggregate score for this scorecard (sum of per-level scores).
        source_url:
          type: string
          format: uri
          description: Link originally supplied in the **OpenScorecardRequest**.
        tags:
          type: array
          description: Arbitrary labels echoed back from the open request.
          items:
            type: string
        user_name:
          type: string
          description: Display name of the user who opened/ran this scorecard.
        user_id:
          type: string
          description: Stable identifier of the user (e.g. provider subject id).
        published_at:
          type: string
          format: date-time
          description: When the scorecard was closed/published (absent if still open).
        environments:
          type: array
          description: >-
            Per-environment breakdown; each entry is one game/environment with
            its runs.
          items:
            $ref: '#/components/schemas/EnvironmentSummary'
        opaque:
          type: object
          description: |
            Free-form JSON blob (≤ 16 KB) exactly as provided when the
            scorecard was opened. Absent if none was supplied.
          additionalProperties: true
        tags_scores:
          type: array
          description: Per-tag aggregate statistics for runs that were tagged.
          items:
            $ref: '#/components/schemas/TagScore'
        open_at:
          type: string
          format: date-time
          description: When the scorecard was opened.
        last_update:
          type: string
          format: date-time
          description: When the scorecard was last updated (e.g. last action or close).
        total_environments_completed:
          type: integer
          description: >-
            Number of environments that reached a terminal state (WIN or
            GAME_OVER).
        total_environments:
          type: integer
          description: Total number of environments in this scorecard.
        total_levels_completed:
          type: integer
          description: Cumulative levels completed across all runs.
        total_levels:
          type: integer
          description: Total number of levels across all environments.
        total_actions:
          type: integer
          description: Cumulative number of actions taken across all plays.
      required:
        - card_id
        - score
        - environments
        - open_at
        - last_update
        - total_environments_completed
        - total_environments
        - total_levels_completed
        - total_levels
        - total_actions
    EnvironmentSummary:
      type: object
      description: |
        Statistics for one environment (game) inside a scorecard.  
        Contains aggregate counts and an array of `runs` (one per RESET/play).
      properties:
        id:
          type: string
          description: Environment/game identifier (e.g. `am92-80effacb`).
        runs:
          type: array
          description: One entry per run (RESET) in this environment.
          items:
            $ref: '#/components/schemas/RunSummary'
        score:
          type: integer
          description: Aggregate score for this environment.
        actions:
          type: integer
          description: Total actions taken in this environment across all runs.
        levels_completed:
          type: integer
          description: Levels completed in this environment.
        completed:
          type: boolean
          description: >-
            Whether this environment reached a terminal state (WIN or
            GAME_OVER).
        level_count:
          type: integer
          description: Number of levels in this environment.
        resets:
          type: integer
          description: Number of RESETs (level or full) in this environment.
      required:
        - id
        - runs
        - score
        - actions
        - levels_completed
        - completed
        - level_count
        - resets
    TagScore:
      type: object
      description: >-
        Per-tag aggregate statistics for runs that were tagged (e.g. for
        filtering).
      properties:
        id:
          type: string
          description: Tag or run identifier.
        guid:
          type: string
          description: Session id associated with this tag entry.
        score:
          type: integer
          description: Aggregate score for this tag.
        levels_completed:
          type: integer
          description: Levels completed for this tag.
        actions:
          type: integer
          description: Total actions for this tag.
        resets:
          type: integer
          description: Resets for this tag.
        state:
          type: string
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
          description: Terminal state for this tag run, if applicable.
        completed:
          type: boolean
          description: Whether this tag run reached a terminal state.
        number_of_levels:
          type: integer
          description: Number of levels.
        number_of_environments:
          type: integer
          description: Number of environments.
      required:
        - id
        - guid
        - score
        - levels_completed
        - actions
        - resets
        - state
        - completed
        - number_of_levels
        - number_of_environments
    RunSummary:
      type: object
      description: >
        Statistics for a single run (one RESET → play until WIN/GAME_OVER or
        abandon)

        within an environment. Arrays `level_scores`, `level_actions`, and

        `level_baseline_actions` align by index (one entry per level).
      properties:
        id:
          type: string
          description: Environment id this run belongs to.
        guid:
          type: string
          description: Server-generated session id for this run.
        score:
          type: integer
          description: Score achieved in this run (0–254).
        levels_completed:
          type: integer
          description: Number of levels completed in this run.
        actions:
          type: integer
          description: Number of actions taken in this run.
        resets:
          type: integer
          description: Number of resets (level or full) in this run.
        state:
          type: string
          description: >
            Final state of the run:

            • **NOT_FINISHED** - run is active.  

            • **NOT_STARTED**  - run has ended and would need RESET to
            continue.  

            • **WIN**          - run ended in victory.  

            • **GAME_OVER**    - run ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        completed:
          type: boolean
          description: Whether the run reached a terminal state (WIN or GAME_OVER).
        level_scores:
          type: array
          description: Score achieved at each level (positional).
          items:
            type: integer
        level_actions:
          type: array
          description: Actions taken at each level (positional).
          items:
            type: integer
        level_baseline_actions:
          type: array
          description: Baseline (e.g. par) actions per level, when defined (positional).
          items:
            type: integer
        number_of_levels:
          type: integer
          description: Number of levels in this environment (may be 0 if not applicable).
        number_of_environments:
          type: integer
          description: Number of environments (may be 0 if not applicable).
      required:
        - id
        - guid
        - score
        - levels_completed
        - actions
        - resets
        - state
        - completed
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).



> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Retrieve scorecard

> Returns the current (or final) statistics for the specified
scorecard.  
This works for both open and already-closed scorecards, making it
useful for polling progress or fetching archived results.




## OpenAPI

````yaml /arc3v1.yaml get /api/scorecard/{card_id}
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/scorecard/{card_id}:
    get:
      tags:
        - Scorecards
      summary: Retrieve scorecard
      description: |
        Returns the current (or final) statistics for the specified
        scorecard.  
        This works for both open and already-closed scorecards, making it
        useful for polling progress or fetching archived results.
      operationId: getScorecard
      parameters:
        - name: card_id
          in: path
          required: true
          description: Identifier returned by **/scorecard/open**.
          schema:
            type: string
      responses:
        '200':
          description: scorecard found; summary returned.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ScorecardSummary'
              examples:
                example:
                  value:
                    card_id: 8ae52f21-b40c-457b-9f4e-65bd8381e67f
                    score: 100
                    source_url: https://sandbox.internal.arc-prize.com
                    tags:
                      - human
                    user_name: distracted_poincare
                    user_id: 102214435702678430912@google
                    open_at: '2026-01-26T23:34:18.213873Z'
                    last_update: '2026-01-26T23:34:53.229182Z'
                    total_environments_completed: 1
                    total_environments: 1
                    total_levels_completed: 5
                    total_levels: 5
                    total_actions: 136
                    environments: []
                    tags_scores: []
        '401':
          description: Missing or invalid **X-API-Key** header.
        '404':
          description: No open or closed scorecard found with the supplied `card_id`.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    ScorecardSummary:
      type: object
      description: >
        Aggregate results for an entire scorecard run.  

        Returned when closing a scorecard or when retrieving a scorecard (open
        or closed).

        Includes cumulative totals, optional metadata echoed from the open
        request

        (e.g. `source_url`, `tags`, `opaque`), user identity (`user_name`,
        `user_id`),

        timestamps (`open_at`, `last_update`, `published_at`), and a
        per-environment

        breakdown in `environments`. The `tags_scores` array provides per-tag
        aggregates

        for runs that were tagged.
      properties:
        card_id:
          type: string
          description: The scorecard ID returned by **OpenScorecardResponse**.
        score:
          type: integer
          description: Aggregate score for this scorecard (sum of per-level scores).
        source_url:
          type: string
          format: uri
          description: Link originally supplied in the **OpenScorecardRequest**.
        tags:
          type: array
          description: Arbitrary labels echoed back from the open request.
          items:
            type: string
        user_name:
          type: string
          description: Display name of the user who opened/ran this scorecard.
        user_id:
          type: string
          description: Stable identifier of the user (e.g. provider subject id).
        published_at:
          type: string
          format: date-time
          description: When the scorecard was closed/published (absent if still open).
        environments:
          type: array
          description: >-
            Per-environment breakdown; each entry is one game/environment with
            its runs.
          items:
            $ref: '#/components/schemas/EnvironmentSummary'
        opaque:
          type: object
          description: |
            Free-form JSON blob (≤ 16 KB) exactly as provided when the
            scorecard was opened. Absent if none was supplied.
          additionalProperties: true
        tags_scores:
          type: array
          description: Per-tag aggregate statistics for runs that were tagged.
          items:
            $ref: '#/components/schemas/TagScore'
        open_at:
          type: string
          format: date-time
          description: When the scorecard was opened.
        last_update:
          type: string
          format: date-time
          description: When the scorecard was last updated (e.g. last action or close).
        total_environments_completed:
          type: integer
          description: >-
            Number of environments that reached a terminal state (WIN or
            GAME_OVER).
        total_environments:
          type: integer
          description: Total number of environments in this scorecard.
        total_levels_completed:
          type: integer
          description: Cumulative levels completed across all runs.
        total_levels:
          type: integer
          description: Total number of levels across all environments.
        total_actions:
          type: integer
          description: Cumulative number of actions taken across all plays.
      required:
        - card_id
        - score
        - environments
        - open_at
        - last_update
        - total_environments_completed
        - total_environments
        - total_levels_completed
        - total_levels
        - total_actions
    EnvironmentSummary:
      type: object
      description: |
        Statistics for one environment (game) inside a scorecard.  
        Contains aggregate counts and an array of `runs` (one per RESET/play).
      properties:
        id:
          type: string
          description: Environment/game identifier (e.g. `am92-80effacb`).
        runs:
          type: array
          description: One entry per run (RESET) in this environment.
          items:
            $ref: '#/components/schemas/RunSummary'
        score:
          type: integer
          description: Aggregate score for this environment.
        actions:
          type: integer
          description: Total actions taken in this environment across all runs.
        levels_completed:
          type: integer
          description: Levels completed in this environment.
        completed:
          type: boolean
          description: >-
            Whether this environment reached a terminal state (WIN or
            GAME_OVER).
        level_count:
          type: integer
          description: Number of levels in this environment.
        resets:
          type: integer
          description: Number of RESETs (level or full) in this environment.
      required:
        - id
        - runs
        - score
        - actions
        - levels_completed
        - completed
        - level_count
        - resets
    TagScore:
      type: object
      description: >-
        Per-tag aggregate statistics for runs that were tagged (e.g. for
        filtering).
      properties:
        id:
          type: string
          description: Tag or run identifier.
        guid:
          type: string
          description: Session id associated with this tag entry.
        score:
          type: integer
          description: Aggregate score for this tag.
        levels_completed:
          type: integer
          description: Levels completed for this tag.
        actions:
          type: integer
          description: Total actions for this tag.
        resets:
          type: integer
          description: Resets for this tag.
        state:
          type: string
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
          description: Terminal state for this tag run, if applicable.
        completed:
          type: boolean
          description: Whether this tag run reached a terminal state.
        number_of_levels:
          type: integer
          description: Number of levels.
        number_of_environments:
          type: integer
          description: Number of environments.
      required:
        - id
        - guid
        - score
        - levels_completed
        - actions
        - resets
        - state
        - completed
        - number_of_levels
        - number_of_environments
    RunSummary:
      type: object
      description: >
        Statistics for a single run (one RESET → play until WIN/GAME_OVER or
        abandon)

        within an environment. Arrays `level_scores`, `level_actions`, and

        `level_baseline_actions` align by index (one entry per level).
      properties:
        id:
          type: string
          description: Environment id this run belongs to.
        guid:
          type: string
          description: Server-generated session id for this run.
        score:
          type: integer
          description: Score achieved in this run (0–254).
        levels_completed:
          type: integer
          description: Number of levels completed in this run.
        actions:
          type: integer
          description: Number of actions taken in this run.
        resets:
          type: integer
          description: Number of resets (level or full) in this run.
        state:
          type: string
          description: >
            Final state of the run:

            • **NOT_FINISHED** - run is active.  

            • **NOT_STARTED**  - run has ended and would need RESET to
            continue.  

            • **WIN**          - run ended in victory.  

            • **GAME_OVER**    - run ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        completed:
          type: boolean
          description: Whether the run reached a terminal state (WIN or GAME_OVER).
        level_scores:
          type: array
          description: Score achieved at each level (positional).
          items:
            type: integer
        level_actions:
          type: array
          description: Actions taken at each level (positional).
          items:
            type: integer
        level_baseline_actions:
          type: array
          description: Baseline (e.g. par) actions per level, when defined (positional).
          items:
            type: integer
        number_of_levels:
          type: integer
          description: Number of levels in this environment (may be 0 if not applicable).
        number_of_environments:
          type: integer
          description: Number of environments (may be 0 if not applicable).
      required:
        - id
        - guid
        - score
        - levels_completed
        - actions
        - resets
        - state
        - completed
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).




> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Retrieve scorecard (one game)

> Returns the scorecard statistics **limited to a single environment**.
Only the entry matching `game_id` is present in `environments`; all
top-level counters are recomputed for that environment alone.

Useful for dashboards that present per-game progress without
fetching the full scorecard payload.




## OpenAPI

````yaml /arc3v1.yaml get /api/scorecard/{card_id}/{game_id}
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/scorecard/{card_id}/{game_id}:
    get:
      tags:
        - Scorecards
      summary: Retrieve scorecard (one game)
      description: |
        Returns the scorecard statistics **limited to a single environment**.
        Only the entry matching `game_id` is present in `environments`; all
        top-level counters are recomputed for that environment alone.

        Useful for dashboards that present per-game progress without
        fetching the full scorecard payload.
      operationId: getScorecardForGame
      parameters:
        - name: card_id
          in: path
          required: true
          description: Identifier returned by **/scorecard/open**.
          schema:
            type: string
        - name: game_id
          in: path
          required: true
          description: Game identifier to filter by (e.g. `ls20-1d57d6daeb05`).
          schema:
            type: string
      responses:
        '200':
          description: scorecard found; statistics for the requested game.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ScorecardSummary'
              examples:
                example:
                  value:
                    card_id: 8ae52f21-b40c-457b-9f4e-65bd8381e67f
                    score: 100
                    source_url: https://sandbox.internal.arc-prize.com
                    tags:
                      - human
                    user_name: distracted_poincare
                    user_id: 102214435702678430912@google
                    open_at: '2026-01-26T23:34:18.213873Z'
                    last_update: '2026-01-26T23:34:53.229182Z'
                    total_environments_completed: 1
                    total_environments: 1
                    total_levels_completed: 5
                    total_levels: 5
                    total_actions: 136
                    environments:
                      - id: ft09-1d57d6daeb05
                        runs:
                          - id: ft09-1d57d6daeb05
                            guid: 66fe6b58-adf8-45b1-ac73-70ed486e22d0
                            score: 3
                            levels_completed: 2
                            actions: 478
                            resets: 0
                            state: WIN
                            completed: true
                            level_scores:
                              - 1
                              - 2
                            level_actions:
                              - 171
                              - 307
                            level_baseline_actions:
                              - 200
                              - 250
                            number_of_levels: 5
                            number_of_environments: 1
                        score: 3
                        actions: 478
                        levels_completed: 2
                        completed: true
                        level_count: 5
                        resets: 0
                    tags_scores: []
        '401':
          description: Missing or invalid **X-API-Key** header.
        '404':
          description: |
            Either the supplied `card_id` does not exist, or the
            scorecard contains no entry for the specified `game_id`.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    ScorecardSummary:
      type: object
      description: >
        Aggregate results for an entire scorecard run.  

        Returned when closing a scorecard or when retrieving a scorecard (open
        or closed).

        Includes cumulative totals, optional metadata echoed from the open
        request

        (e.g. `source_url`, `tags`, `opaque`), user identity (`user_name`,
        `user_id`),

        timestamps (`open_at`, `last_update`, `published_at`), and a
        per-environment

        breakdown in `environments`. The `tags_scores` array provides per-tag
        aggregates

        for runs that were tagged.
      properties:
        card_id:
          type: string
          description: The scorecard ID returned by **OpenScorecardResponse**.
        score:
          type: integer
          description: Aggregate score for this scorecard (sum of per-level scores).
        source_url:
          type: string
          format: uri
          description: Link originally supplied in the **OpenScorecardRequest**.
        tags:
          type: array
          description: Arbitrary labels echoed back from the open request.
          items:
            type: string
        user_name:
          type: string
          description: Display name of the user who opened/ran this scorecard.
        user_id:
          type: string
          description: Stable identifier of the user (e.g. provider subject id).
        published_at:
          type: string
          format: date-time
          description: When the scorecard was closed/published (absent if still open).
        environments:
          type: array
          description: >-
            Per-environment breakdown; each entry is one game/environment with
            its runs.
          items:
            $ref: '#/components/schemas/EnvironmentSummary'
        opaque:
          type: object
          description: |
            Free-form JSON blob (≤ 16 KB) exactly as provided when the
            scorecard was opened. Absent if none was supplied.
          additionalProperties: true
        tags_scores:
          type: array
          description: Per-tag aggregate statistics for runs that were tagged.
          items:
            $ref: '#/components/schemas/TagScore'
        open_at:
          type: string
          format: date-time
          description: When the scorecard was opened.
        last_update:
          type: string
          format: date-time
          description: When the scorecard was last updated (e.g. last action or close).
        total_environments_completed:
          type: integer
          description: >-
            Number of environments that reached a terminal state (WIN or
            GAME_OVER).
        total_environments:
          type: integer
          description: Total number of environments in this scorecard.
        total_levels_completed:
          type: integer
          description: Cumulative levels completed across all runs.
        total_levels:
          type: integer
          description: Total number of levels across all environments.
        total_actions:
          type: integer
          description: Cumulative number of actions taken across all plays.
      required:
        - card_id
        - score
        - environments
        - open_at
        - last_update
        - total_environments_completed
        - total_environments
        - total_levels_completed
        - total_levels
        - total_actions
    EnvironmentSummary:
      type: object
      description: |
        Statistics for one environment (game) inside a scorecard.  
        Contains aggregate counts and an array of `runs` (one per RESET/play).
      properties:
        id:
          type: string
          description: Environment/game identifier (e.g. `am92-80effacb`).
        runs:
          type: array
          description: One entry per run (RESET) in this environment.
          items:
            $ref: '#/components/schemas/RunSummary'
        score:
          type: integer
          description: Aggregate score for this environment.
        actions:
          type: integer
          description: Total actions taken in this environment across all runs.
        levels_completed:
          type: integer
          description: Levels completed in this environment.
        completed:
          type: boolean
          description: >-
            Whether this environment reached a terminal state (WIN or
            GAME_OVER).
        level_count:
          type: integer
          description: Number of levels in this environment.
        resets:
          type: integer
          description: Number of RESETs (level or full) in this environment.
      required:
        - id
        - runs
        - score
        - actions
        - levels_completed
        - completed
        - level_count
        - resets
    TagScore:
      type: object
      description: >-
        Per-tag aggregate statistics for runs that were tagged (e.g. for
        filtering).
      properties:
        id:
          type: string
          description: Tag or run identifier.
        guid:
          type: string
          description: Session id associated with this tag entry.
        score:
          type: integer
          description: Aggregate score for this tag.
        levels_completed:
          type: integer
          description: Levels completed for this tag.
        actions:
          type: integer
          description: Total actions for this tag.
        resets:
          type: integer
          description: Resets for this tag.
        state:
          type: string
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
          description: Terminal state for this tag run, if applicable.
        completed:
          type: boolean
          description: Whether this tag run reached a terminal state.
        number_of_levels:
          type: integer
          description: Number of levels.
        number_of_environments:
          type: integer
          description: Number of environments.
      required:
        - id
        - guid
        - score
        - levels_completed
        - actions
        - resets
        - state
        - completed
        - number_of_levels
        - number_of_environments
    RunSummary:
      type: object
      description: >
        Statistics for a single run (one RESET → play until WIN/GAME_OVER or
        abandon)

        within an environment. Arrays `level_scores`, `level_actions`, and

        `level_baseline_actions` align by index (one entry per level).
      properties:
        id:
          type: string
          description: Environment id this run belongs to.
        guid:
          type: string
          description: Server-generated session id for this run.
        score:
          type: integer
          description: Score achieved in this run (0–254).
        levels_completed:
          type: integer
          description: Number of levels completed in this run.
        actions:
          type: integer
          description: Number of actions taken in this run.
        resets:
          type: integer
          description: Number of resets (level or full) in this run.
        state:
          type: string
          description: >
            Final state of the run:

            • **NOT_FINISHED** - run is active.  

            • **NOT_STARTED**  - run has ended and would need RESET to
            continue.  

            • **WIN**          - run ended in victory.  

            • **GAME_OVER**    - run ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        completed:
          type: boolean
          description: Whether the run reached a terminal state (WIN or GAME_OVER).
        level_scores:
          type: array
          description: Score achieved at each level (positional).
          items:
            type: integer
        level_actions:
          type: array
          description: Actions taken at each level (positional).
          items:
            type: integer
        level_baseline_actions:
          type: array
          description: Baseline (e.g. par) actions per level, when defined (positional).
          items:
            type: integer
        number_of_levels:
          type: integer
          description: Number of levels in this environment (may be 0 if not applicable).
        number_of_environments:
          type: integer
          description: Number of environments (may be 0 if not applicable).
      required:
        - id
        - guid
        - score
        - levels_completed
        - actions
        - resets
        - state
        - completed
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Start or reset game instance

> Creates a new game session **or** resets an existing one,
depending on the presence of `guid` in the request body:

• **Omit `guid` or set it to `null`** → start a brand-new game
  instance.  
• **Provide an existing `guid`** → reset that session.  
  - If at least one ACTION command has been issued since the last
    level transition, only the **current level** is restarted.  
  - If no ACTIONs have been issued, the entire game resets.  
  Two consecutive RESETs therefore guarantee a completely fresh
  game.

The call always returns the first (or refreshed) frame of the
game state, along with updated score and win condition.

**Note:** The response includes cookies (particularly `AWSALB*` cookies) that must be included in all subsequent ACTION commands for this session. These cookies ensure requests are routed to the same backend instance maintaining your game state.




## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/RESET
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/RESET:
    post:
      tags:
        - Commands
      summary: Start or reset game instance
      description: >
        Creates a new game session **or** resets an existing one,

        depending on the presence of `guid` in the request body:


        • **Omit `guid` or set it to `null`** → start a brand-new game
          instance.  
        • **Provide an existing `guid`** → reset that session.  
          - If at least one ACTION command has been issued since the last
            level transition, only the **current level** is restarted.  
          - If no ACTIONs have been issued, the entire game resets.  
          Two consecutive RESETs therefore guarantee a completely fresh
          game.

        The call always returns the first (or refreshed) frame of the

        game state, along with updated score and win condition.


        **Note:** The response includes cookies (particularly `AWSALB*` cookies)
        that must be included in all subsequent ACTION commands for this
        session. These cookies ensure requests are routed to the same backend
        instance maintaining your game state.
      operationId: resetGame
      requestBody:
        description: Game identifier, scorecard ID, and (optionally) the session `guid`.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ResetCommand'
            examples:
              newGame:
                summary: Start a new session
                value:
                  game_id: ls20-016295f7601e
                  card_id: 8bb3b1b8-4b46-4a29-a13b-ad7850a0f916
              levelReset:
                summary: Reset current level of an existing session
                value:
                  game_id: ls20-016295f7601e
                  card_id: 8bb3b1b8-4b46-4a29-a13b-ad7850a0f916
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
      responses:
        '200':
          description: First frame after starting or resetting the session.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - 0
                          - 0
                          - 0
                          - …
                        - - …
                    state: NOT_FINISHED
                    levels_completed: 0
                    win_levels: 254
                    action_input:
                      id: 0
                      data: {}
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id`  
            • Missing or unknown `card_id`  
            • `guid` does not correspond to an active session
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    ResetCommand:
      type: object
      description: |
        Starts a new game session **or** resets an existing one, depending on
        whether a `guid` is supplied.

        • **No `guid` (null/empty)** → A brand-new game instance is created and
          the response will include its freshly minted `guid`.

        • **With `guid`** → The server issues a reset to that specific
          instance:
            - If at least one ACTION command has been executed in the **current
              level**, only that level is reset (typical “try again” behaviour).
            - If no ACTION commands have been executed since the last level
              transition, the entire game is reset to its initial state.

        Sending two RESET commands back-to-back therefore always yields a
        completely fresh game.

        All plays should be associated with an open scorecard via `card_id`
        so aggregated results can be tracked.
      properties:
        game_id:
          type: string
          description: Identifier of the game to start or reset (e.g. `ls20`).
        card_id:
          type: string
          description: |
            scorecard identifier returned by **OpenScorecardResponse**. Required
            to attribute this play to the correct scorecard.
        guid:
          type: string
          nullable: true
          description: |
            Server-generated game session ID.  
            • Omit or set to `null` to create a new game.  
            • Provide an existing value to reset that game as described above.
      required:
        - game_id
        - card_id
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Execute simple action 1

> Issues **ACTION 1** to the specified game session.  
This is a single-parameter command (no X/Y coordinates): the exact
in-game effect depends on the title—for example, it might
represent “move up” or “select option A”.

The request must include:
• `game_id` — which game to act on  
• `guid` — the active session identifier returned from RESET  

An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
audit or research purposes.

A successful call returns the next visual frame(s) and updated
score/state.




## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/ACTION1
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/ACTION1:
    post:
      tags:
        - Commands
      summary: Execute simple action 1
      description: |
        Issues **ACTION 1** to the specified game session.  
        This is a single-parameter command (no X/Y coordinates): the exact
        in-game effect depends on the title—for example, it might
        represent “move up” or “select option A”.

        The request must include:
        • `game_id` — which game to act on  
        • `guid` — the active session identifier returned from RESET  

        An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
        audit or research purposes.

        A successful call returns the next visual frame(s) and updated
        score/state.
      operationId: action1
      requestBody:
        description: Game/session identifiers plus optional reasoning data.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SimpleActionCommand'
            examples:
              action:
                value:
                  game_id: ls20-016295f7601e
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                  reasoning:
                    policy: π_left
      responses:
        '200':
          description: Frame returned after executing the action.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - 0
                          - 0
                          - 1
                          - …
                        - - …
                    state: NOT_FINISHED
                    levels_completed: 3
                    win_levels: 254
                    action_input:
                      id: 1
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id` or invalid format  
            • `guid` not found or does not belong to `game_id`  
            • `reasoning` field exceeds 16 KB or is malformed
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    SimpleActionCommand:
      type: object
      description: >
        Issues a one-parameter action (ACTION1 - ACTION5) to a running

        game instance identified by `guid`.


        **Important:** Include any cookies (especially `AWSALB*` cookies)
        received from previous RESET or ACTION responses to ensure session
        affinity.
      properties:
        game_id:
          type: string
          description: Game identifier this action targets.
        guid:
          type: string
          description: Server-generated session ID obtained from a RESET response.
        reasoning:
          type: object
          description: |
            Optional, caller-defined JSON blob (≤ 16 KB) capturing the
            agent's internal reasoning, model parameters, or any other
            metadata you'd like to store alongside the action.
          additionalProperties: true
      required:
        - game_id
        - guid
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).



> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Execute simple action 2

> Issues **ACTION 2** to the specified game session.  
This is a single-parameter command (no X/Y coordinates): the exact
in-game effect depends on the title—for example, it might
represent “move down" or “select option B”.

The request must include:
• `game_id` — which game to act on  
• `guid` — the active session identifier returned from RESET  

An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
audit or research purposes.

A successful call returns the next visual frame(s) and updated
score/state.




## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/ACTION2
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/ACTION2:
    post:
      tags:
        - Commands
      summary: Execute simple action 2
      description: |
        Issues **ACTION 2** to the specified game session.  
        This is a single-parameter command (no X/Y coordinates): the exact
        in-game effect depends on the title—for example, it might
        represent “move down" or “select option B”.

        The request must include:
        • `game_id` — which game to act on  
        • `guid` — the active session identifier returned from RESET  

        An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
        audit or research purposes.

        A successful call returns the next visual frame(s) and updated
        score/state.
      operationId: action2
      requestBody:
        description: Game/session identifiers plus optional reasoning data.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SimpleActionCommand'
            examples:
              action:
                value:
                  game_id: ls20-016295f7601e
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                  reasoning:
                    policy: π_left
      responses:
        '200':
          description: Frame returned after executing the action.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - 0
                          - 0
                          - 1
                          - …
                        - - …
                    state: NOT_FINISHED
                    levels_completed: 3
                    win_levels: 254
                    action_input:
                      id: 2
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id` or invalid format  
            • `guid` not found or does not belong to `game_id`  
            • `reasoning` field exceeds 16 KB or is malformed
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    SimpleActionCommand:
      type: object
      description: >
        Issues a one-parameter action (ACTION1 - ACTION5) to a running

        game instance identified by `guid`.


        **Important:** Include any cookies (especially `AWSALB*` cookies)
        received from previous RESET or ACTION responses to ensure session
        affinity.
      properties:
        game_id:
          type: string
          description: Game identifier this action targets.
        guid:
          type: string
          description: Server-generated session ID obtained from a RESET response.
        reasoning:
          type: object
          description: |
            Optional, caller-defined JSON blob (≤ 16 KB) capturing the
            agent's internal reasoning, model parameters, or any other
            metadata you'd like to store alongside the action.
          additionalProperties: true
      required:
        - game_id
        - guid
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).



> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Execute simple action 3

> Issues **ACTION 3** to the specified game session.  
This is a single-parameter command (no X/Y coordinates): the exact
in-game effect depends on the title—for example, it might
represent “move left” or “select option C”.

The request must include:
• `game_id` — which game to act on  
• `guid` — the active session identifier returned from RESET  

An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
audit or research purposes.

A successful call returns the next visual frame(s) and updated
score/state.




## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/ACTION3
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/ACTION3:
    post:
      tags:
        - Commands
      summary: Execute simple action 3
      description: |
        Issues **ACTION 3** to the specified game session.  
        This is a single-parameter command (no X/Y coordinates): the exact
        in-game effect depends on the title—for example, it might
        represent “move left” or “select option C”.

        The request must include:
        • `game_id` — which game to act on  
        • `guid` — the active session identifier returned from RESET  

        An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
        audit or research purposes.

        A successful call returns the next visual frame(s) and updated
        score/state.
      operationId: action3
      requestBody:
        description: Game/session identifiers plus optional reasoning data.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SimpleActionCommand'
            examples:
              action:
                value:
                  game_id: ls20-016295f7601e
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                  reasoning:
                    policy: π_left
      responses:
        '200':
          description: Frame returned after executing the action.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - 0
                          - 0
                          - 1
                          - …
                        - - …
                    state: NOT_FINISHED
                    levels_completed: 3
                    win_levels: 254
                    action_input:
                      id: 3
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id` or invalid format  
            • `guid` not found or does not belong to `game_id`  
            • `reasoning` field exceeds 16 KB or is malformed
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    SimpleActionCommand:
      type: object
      description: >
        Issues a one-parameter action (ACTION1 - ACTION5) to a running

        game instance identified by `guid`.


        **Important:** Include any cookies (especially `AWSALB*` cookies)
        received from previous RESET or ACTION responses to ensure session
        affinity.
      properties:
        game_id:
          type: string
          description: Game identifier this action targets.
        guid:
          type: string
          description: Server-generated session ID obtained from a RESET response.
        reasoning:
          type: object
          description: |
            Optional, caller-defined JSON blob (≤ 16 KB) capturing the
            agent's internal reasoning, model parameters, or any other
            metadata you'd like to store alongside the action.
          additionalProperties: true
      required:
        - game_id
        - guid
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).



> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Execute simple action 4

> Issues **ACTION 4** to the specified game session.  
This is a single-parameter command (no X/Y coordinates): the exact
in-game effect depends on the title—for example, it might
represent “move right" or “select option D”.

The request must include:
• `game_id` — which game to act on  
• `guid` — the active session identifier returned from RESET  

An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
audit or research purposes.

A successful call returns the next visual frame(s) and updated
score/state.




## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/ACTION4
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/ACTION4:
    post:
      tags:
        - Commands
      summary: Execute simple action 4
      description: |
        Issues **ACTION 4** to the specified game session.  
        This is a single-parameter command (no X/Y coordinates): the exact
        in-game effect depends on the title—for example, it might
        represent “move right" or “select option D”.

        The request must include:
        • `game_id` — which game to act on  
        • `guid` — the active session identifier returned from RESET  

        An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
        audit or research purposes.

        A successful call returns the next visual frame(s) and updated
        score/state.
      operationId: action4
      requestBody:
        description: Game/session identifiers plus optional reasoning data.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SimpleActionCommand'
            examples:
              action:
                value:
                  game_id: ls20-016295f7601e
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                  reasoning:
                    policy: π_left
      responses:
        '200':
          description: Frame returned after executing the action.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - 0
                          - 0
                          - 1
                          - …
                        - - …
                    state: NOT_FINISHED
                    levels_completed: 3
                    win_levels: 254
                    action_input:
                      id: 4
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id` or invalid format  
            • `guid` not found or does not belong to `game_id`  
            • `reasoning` field exceeds 16 KB or is malformed
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    SimpleActionCommand:
      type: object
      description: >
        Issues a one-parameter action (ACTION1 - ACTION5) to a running

        game instance identified by `guid`.


        **Important:** Include any cookies (especially `AWSALB*` cookies)
        received from previous RESET or ACTION responses to ensure session
        affinity.
      properties:
        game_id:
          type: string
          description: Game identifier this action targets.
        guid:
          type: string
          description: Server-generated session ID obtained from a RESET response.
        reasoning:
          type: object
          description: |
            Optional, caller-defined JSON blob (≤ 16 KB) capturing the
            agent's internal reasoning, model parameters, or any other
            metadata you'd like to store alongside the action.
          additionalProperties: true
      required:
        - game_id
        - guid
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Execute simple action 5

> Issues **ACTION 5** to the specified game session.  
This is a single-parameter command (no X/Y coordinates): the exact
in-game effect depends on the title—for example, it might
represent “jump”, "rotate", "fire" or “select option E”.

The request must include:
• `game_id` — which game to act on  
• `guid` — the active session identifier returned from RESET  

An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
audit or research purposes.

A successful call returns the next visual frame(s) and updated
score/state.




## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/ACTION5
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/ACTION5:
    post:
      tags:
        - Commands
      summary: Execute simple action 5
      description: |
        Issues **ACTION 5** to the specified game session.  
        This is a single-parameter command (no X/Y coordinates): the exact
        in-game effect depends on the title—for example, it might
        represent “jump”, "rotate", "fire" or “select option E”.

        The request must include:
        • `game_id` — which game to act on  
        • `guid` — the active session identifier returned from RESET  

        An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
        audit or research purposes.

        A successful call returns the next visual frame(s) and updated
        score/state.
      operationId: action5
      requestBody:
        description: Game/session identifiers plus optional reasoning data.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SimpleActionCommand'
            examples:
              action:
                value:
                  game_id: ls20-016295f7601e
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                  reasoning:
                    policy: π_left
      responses:
        '200':
          description: Frame returned after executing the action.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - 0
                          - 0
                          - 1
                          - …
                        - - …
                    state: NOT_FINISHED
                    levels_completed: 3
                    win_levels: 254
                    action_input:
                      id: 5
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id` or invalid format  
            • `guid` not found or does not belong to `game_id`  
            • `reasoning` field exceeds 16 KB or is malformed
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    SimpleActionCommand:
      type: object
      description: >
        Issues a one-parameter action (ACTION1 - ACTION5) to a running

        game instance identified by `guid`.


        **Important:** Include any cookies (especially `AWSALB*` cookies)
        received from previous RESET or ACTION responses to ensure session
        affinity.
      properties:
        game_id:
          type: string
          description: Game identifier this action targets.
        guid:
          type: string
          description: Server-generated session ID obtained from a RESET response.
        reasoning:
          type: object
          description: |
            Optional, caller-defined JSON blob (≤ 16 KB) capturing the
            agent's internal reasoning, model parameters, or any other
            metadata you'd like to store alongside the action.
          additionalProperties: true
      required:
        - game_id
        - guid
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).



> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Execute complex action (requires x,y)

> Issues **ACTION 6**—a two-parameter command that supplies explicit
X/Y coordinates—to an active game session.  Common use-cases
include “click/tap at (x,y)”, “place a tile”, or “shoot a
projectile,” depending on the game's mechanics.

Required fields  
• `game_id` — the game to act in  
• `guid`    — session identifier obtained from RESET  
• `x`,`y`   — zero-based grid coordinates (0-63 inclusive)

On success the server applies the action, advances game logic to
the next stable frame, and returns that frame together with the
updated score, state, and win condition.




## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/ACTION6
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/ACTION6:
    post:
      tags:
        - Commands
      summary: Execute complex action (requires x,y)
      description: |
        Issues **ACTION 6**—a two-parameter command that supplies explicit
        X/Y coordinates—to an active game session.  Common use-cases
        include “click/tap at (x,y)”, “place a tile”, or “shoot a
        projectile,” depending on the game's mechanics.

        Required fields  
        • `game_id` — the game to act in  
        • `guid`    — session identifier obtained from RESET  
        • `x`,`y`   — zero-based grid coordinates (0-63 inclusive)

        On success the server applies the action, advances game logic to
        the next stable frame, and returns that frame together with the
        updated score, state, and win condition.
      operationId: action6
      requestBody:
        description: Game/session identifiers plus the coordinate payload.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ComplexActionCommand'
            examples:
              placeTile:
                summary: Place at (12, 34)
                value:
                  game_id: ls20-016295f7601e
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                  x: 12
                  'y': 34
      responses:
        '200':
          description: Frame returned after executing the action.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - …
                    state: NOT_FINISHED
                    levels_completed: 17
                    win_levels: 254
                    action_input:
                      id: 6
                      data:
                        x: 12
                        'y': 34
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id`  
            • `guid` not found or does not belong to the supplied `game_id`  
            • `x` or `y` outside the 0-63 range
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    ComplexActionCommand:
      type: object
      description: >
        Payload for coordinate-based actions (e.g. `/api/cmd/ACTION6`).

        Supplies an `(x, y)` location on the 64 × 64 game grid along with

        the game/session identifiers so the engine can apply the action

        to the correct running instance.


        **Important:** Include any cookies (especially `AWSALB*` cookies)
        received from previous RESET or ACTION responses to ensure session
        affinity.
      properties:
        game_id:
          type: string
          description: Identifier of the game receiving this action.
        guid:
          type: string
          description: Server-generated session ID obtained from the RESET call.
        x:
          type: integer
          minimum: 0
          maximum: 63
          description: Horizontal coordinate on the game grid (0 = left, 63 = right).
        'y':
          type: integer
          minimum: 0
          maximum: 63
          description: Vertical coordinate on the game grid (0 = top, 63 = bottom).
        reasoning:
          type: object
          description: |
            Optional, caller-defined JSON blob (≤ 16 KB) capturing the
            agent's internal reasoning, model parameters, or any other
            metadata you'd like to store alongside the action.
          additionalProperties: true
      required:
        - game_id
        - guid
        - x
        - 'y'
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).




> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Execute simple action 7

> Issues **ACTION 7** to the specified game session.  
This is a single-parameter command (no X/Y coordinates): 
ACTION7 will always be an undo action for games that support it.

The request must include:
• `game_id` — which game to act on  
• `guid` — the active session identifier returned from RESET  

An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
audit or research purposes.

A successful call returns the next visual frame(s) and updated
score/state.




## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/ACTION7
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/ACTION7:
    post:
      tags:
        - Commands
      summary: Execute simple action 7
      description: |
        Issues **ACTION 7** to the specified game session.  
        This is a single-parameter command (no X/Y coordinates): 
        ACTION7 will always be an undo action for games that support it.

        The request must include:
        • `game_id` — which game to act on  
        • `guid` — the active session identifier returned from RESET  

        An optional `reasoning` JSON blob (≤ 16 KB) can be attached for
        audit or research purposes.

        A successful call returns the next visual frame(s) and updated
        score/state.
      operationId: action7
      requestBody:
        description: Game/session identifiers plus optional reasoning data.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SimpleActionCommand'
            examples:
              action:
                value:
                  game_id: ls20-016295f7601e
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                  reasoning:
                    policy: π_left
      responses:
        '200':
          description: Frame returned after executing the action.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - 0
                          - 0
                          - 1
                          - …
                        - - …
                    state: NOT_FINISHED
                    levels_completed: 3
                    win_levels: 254
                    action_input:
                      id: 7
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
                      - 7
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id` or invalid format  
            • `guid` not found or does not belong to `game_id`  
            • `reasoning` field exceeds 16 KB or is malformed
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    SimpleActionCommand:
      type: object
      description: >
        Issues a one-parameter action (ACTION1 - ACTION5) to a running

        game instance identified by `guid`.


        **Important:** Include any cookies (especially `AWSALB*` cookies)
        received from previous RESET or ACTION responses to ensure session
        affinity.
      properties:
        game_id:
          type: string
          description: Game identifier this action targets.
        guid:
          type: string
          description: Server-generated session ID obtained from a RESET response.
        reasoning:
          type: object
          description: |
            Optional, caller-defined JSON blob (≤ 16 KB) capturing the
            agent's internal reasoning, model parameters, or any other
            metadata you'd like to store alongside the action.
          additionalProperties: true
      required:
        - game_id
        - guid
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````

Built with [Mintlify](https://mintlify.com).



BASED ON THE DOCS DOES IT FOLLOW THE BENCHMARK AND SUBMISSIONS OR THATS SHOULD BE DIFFERENT? BELIV THE TOOLKIT IS ALREAY IN THE ROOT HERE: arc-agi-training