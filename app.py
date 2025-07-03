import gradio as gr
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Load XGBoost model and test data ---
with open("models/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
X_test = pd.read_csv("data/X_test.csv")

# --- Load fine-tuned Falcon model ---
tokenizer = AutoTokenizer.from_pretrained("models/falcon-finetuned")
model = AutoModelForCausalLM.from_pretrained("models/falcon-finetuned", device_map="auto", torch_dtype=torch.float16)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

# --- Failure Reason Map ---
failure_reason_map = {
    "bearings": ["noise_db", "gaccx", "gaccy", "gaccz", "oil_tank_temp"],
    "wpump": ["wpump_power", "water_flow", "water_inlet_temp", "water_outlet_temp"],
    "radiator": ["water_inlet_temp", "water_outlet_temp", "wpump_outlet_press"],
    "exvalve": ["haccx", "haccy", "haccz", "air_flow", "outlet_pressure_bar"]
}

def trace_risk_reason(row, model, top_n=5):
    features = row.index.tolist()
    values = row.values.flatten()
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_n]
    top_feats = [(features[i], values[i]) for i in top_indices]
    explanation = []
    for feat, val in top_feats:
        for comp, relevant_feats in failure_reason_map.items():
            if feat in relevant_feats:
                explanation.append(f"{feat} is high ({val:.2f}), suggesting possible issue in {comp}")
                break
    return explanation

def generate_llm_prompt(row):
    pred_class = xgb_model.predict(row.values.reshape(1, -1))[0]
    explanation_lines = trace_risk_reason(row, xgb_model)
    prompt = f"""This compressor is predicted to be at RISK LEVEL {pred_class}.

Possible reasons:\n"""
    for line in explanation_lines:
        prompt += f"- {line}\n"
    prompt += "\nPlease explain this risk level and what it entails (possible reasons)."
    return prompt

def bar_plot(row, idx):
    fig, ax = plt.subplots(figsize=(10, 4))
    row.plot(kind="bar", ax=ax, color="orange")
    ax.set_ylabel("Sensor Reading")
    ax.set_title(f"Sensor Values for Compressor #{idx}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def box_plot(row, idx):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=X_test, orient="h", ax=ax, color="lightgray")
    for i, val in enumerate(row.values):
        ax.plot(val, i, "ro")
    ax.set_title(f"Boxplot for Compressor #{idx}")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def radar_plot(row, idx):
    categories = list(row.index)
    values = row.values.tolist()
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='orange', linewidth=2)
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_title(f'Radar Plot: Compressor #{idx}')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def diagnose_compressor(idx, plot_type="Bar"):
    idx = int(idx)
    row = X_test.iloc[idx]
    pred_risk = int(xgb_model.predict(row.values.reshape(1, -1))[0])
    reasons = "\n".join(trace_risk_reason(row, xgb_model))

    if plot_type == "Bar":
        img = bar_plot(row, idx)
    elif plot_type == "Boxplot":
        img = box_plot(row, idx)
    else:
        img = radar_plot(row, idx)

    prompt = generate_llm_prompt(row)
    llm_response = llm(prompt)[0]['generated_text']

    return pred_risk, reasons, img, llm_response




with gr.Blocks() as demo:
    gr.Markdown("Air Compressor Diagnostics with LLM Explanation")

    with gr.Row():
        idx_input = gr.Number(label="Compressor Index(0-299)", value=0, precision=0)
        plot_type = gr.Radio(["Bar", "Boxplot", "Radar"], value="Bar", label="Choose Visualization Type")

    run_btn = gr.Button("Run Diagnosis")

    risk_output = gr.Number(label="Predicted Risk Level", precision=0)
    reason_output = gr.Textbox(label="Top Contributing Features")
    image_output = gr.Image(type="pil", label="Sensor Visualization")
    llm_output = gr.Textbox(label="Technician-friendly Explanation", lines=10)

    run_btn.click(fn=diagnose_compressor, inputs=[idx_input, plot_type],
                  outputs=[risk_output, reason_output, image_output, llm_output])

    
demo.launch()
