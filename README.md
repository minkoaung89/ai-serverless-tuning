###AI-Driven Auto-Tuning for Serverless Performance Optimization##

This project implements a hybrid **Machine Learning + Reinforcement Learning (ML+RL)** 
framework to dynamically optimize the runtime configurations (memory, timeout, concurrency) 
of serverless functions (e.g., AWS Lambda). The system leverages **ML-based workload forecasting** 
and **RL-based configuration tuning** to improve latency, reduce SLA violations, and 
maintain cost efficiency in cloud-native deployments.

-------------------------------------------------------------------------------------------------------

### Poject Highlights

- Forecasts workloads using XGBoost ML model
- Dynamically tunes Lambda configuration using A2C RL agent
- Closed-loop auto-tuning architecture with feedback integration
- Synthetic workload generation with bursty, periodic, and irregular traffic patterns
- Evaluated across five experimental scenarios
- Includes AWS Lambda proof-of-concept (PoC)

-------------------------------------------------------------------------------------------------------

### Project Structure

```bash
.
├── serverless_tuning_colab.ipynb        #Google Colab notebook (main logic)
├── requirements.txt                     #Python dependencies
├── architecture.png                     #System architecture diagram
├── data/
│   ├── a2c_serverless_model.zip          #Trained A2C RL agent (zipped model)
│   ├── xgb_best_model.pkl                #Trained XGBoost workload predictor
│   ├── synthetic_invocations.csv         #Input workload sequence for testing
│   ├── baseline_hybrid.csv               #ML+RL (Hybrid) strategy results
│   ├── baseline_hybrid_a2c.csv           #ML+RL A2C-specific logs
│   ├── baseline_ml_only.csv              #ML-only strategy results
│   ├── baseline_ml_only_a2c.csv          #ML-only with A2C adjustments
│   ├── baseline_rl_only.csv              #RL-only strategy results
│   ├── baseline_rl_only_a2c.csv          #RL-only with A2C details
│   ├── baseline_static.csv               #Static configuration baseline
│   ├── baseline_static_a2c.csv           #Static with A2C recorded logs
├── src/
│   └── env.py                           #Custom Gym environment
│   └── generate_workload.py             #Workload generator for synthetic traffic types
├── README.md
├── LICENSE
├── aws_poc/
│   ├── lambda_function.py          		#Minimal Lambda function
│   ├── cloudwatch_metric_fetch.py          #Script that fetch cloudwatch metrics to validate 
│   ├── README.md         					#Instructions of how to implement AWS_POC

-------------------------------------------------------------------------------------------------------

### How to Install & Run the Notebook

### Option A — Google Colab (recommended)
1. Open **serverless_tuning_colab.ipynb** in Colab (File → Open notebook → GitHub → paste this repo URL).
2. *(Optional)* Runtime → **Change runtime type** → **GPU: T4**.
3. Run the first setup cell in the notebook to install dependencies (uses `requirements.txt`).
4. **Artifacts are preloaded:** trained models, synthetic workloads, and baseline logs are already in `data/`. 
Running the notebook will **reuse** these to reproduce results. Training/regeneration cells are optional and may overwrite files.


### Option B — Run Locally (Jupyter)
**Prereqs:** Python 3.10+, Jupyter Notebook/Lab.
```bash
# from the repo root
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook  # or: jupyter lab
