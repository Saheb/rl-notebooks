# 🧠 RL Notebooks

**A 16-notebook journey through Reinforcement Learning — from Bellman to AlphaZero to RLHF.**

Learn by doing: each notebook is a workbook with guided `# TODO` scaffolds you fill in yourself. Solutions are included so you can check your work.

## 📖 Structure

| # | Title | Key Ideas |
|---|-------|-----------|
| **Act I — Mathematical Foundations** | | |
| 01 | The Game of Life | Agents, Environments, MDPs |
| 02 | Time Travel | Returns, Bellman Equation |
| 03 | The Spreadsheet of the Mind | Dynamic Programming, Value Iteration |
| **Act II — Value-Based** | | |
| 04 | Learning by Stumbling | Q-Learning, Explore/Exploit |
| 05 | Giving the AI Eyes | DQN, Replay Buffers |
| 06 | Brain Hacks | Double & Dueling DQN |
| **Act III — Policy-Based** | | |
| 07 | Throwing Away the Table | Policy Gradients, REINFORCE |
| 08 | The Player and the Coach | Actor-Critic |
| 09 | Stepping Carefully | PPO |
| **Act IV — Engineering** | | |
| 10 | The Steering Wheel | Continuous Actions, PPO |
| 11 | The Clone Army | Distributed RL, A3C/IMPALA |
| 12 | Learning in the Dark | Offline RL, CQL |
| **Act V — LLM Alignment** | | |
| 13 | Slaying the Memory Monster | GRPO (DeepSeek) |
| 14 | The Great Bypass | DPO |
| **Act VI — Grandmasters** | | |
| 15 | The Infinite Curriculum | Self-Play, MCTS, AlphaZero |
| 16 | The Hidden Board | CFR, Poker |

## 📂 Folders

- **`notebooks/`** — Complete notebooks with solutions filled in
- **`exercises/`** — Same notebooks with `# TODO` blocks for you to complete
- **`utils/`** — Shared plotting and environment helpers

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rl-notebooks.git
cd rl-notebooks

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Start with `exercises/01_agents_envs_mdps.ipynb` if you want to learn by doing, or `notebooks/01_agents_envs_mdps.ipynb` if you want to read the completed version.

## 🎯 Prerequisites

- Python 3.10+
- Basic comfort with NumPy
- High-school math (we introduce all the RL math gently)
- Curiosity 🙂

## ☁️ Google Colab

Every notebook works in Colab — click the badge at the top of each notebook, or upload the `.ipynb` file directly.

## 📝 License

MIT
