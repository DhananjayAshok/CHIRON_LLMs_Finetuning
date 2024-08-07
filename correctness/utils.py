from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt
import torch

class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto")
    
    def predict(self, inp):
        inputs = self.tokenizer(inp, return_tensors="pt")
        logits = self.model(**inputs).logits
        logits = logits.detach().numpy()
        softmaxed = np.exp(logits) / np.sum(np.exp(logits))
        return softmaxed[0, 1]
    
    def __str__(self):
        return self.model_path.split("/")[-1]
    
    def __repr__(self):
        return self.model_path.split("/")[-1]


class Baselines:
    def __init__(self, baseline_name="coin"):
        self.baseline_name = baseline_name
        assert baseline_name in ["coin", "uniform", "label_prior", "constant_1", "constant_0"]

    def predict(self, inp):
        if self.baseline_name == "coin":
            # 50% chance of 0, 50% chance of 1
            return np.random.choice([0, 1])
        elif self.baseline_name == "uniform":
            return np.random.uniform()
        elif self.baseline_name == "label_prior":
            # 92% chance of false, 8% chance of true
            return np.random.choice([0, 1], p=[0.92, 0.08])
        elif self.baseline_name == "constant_1":
            return 1
        elif self.baseline_name == "constant_0":
            return 0
        
    def __str__(self):
        return self.baseline_name
    
    def __repr__(self):
        return self.baseline_name


def predict(df_path, method):
    outpath = df_path.replace(".csv", f"{str(method)}_predicted.csv")
    if os.path.exists(outpath):
        print(f"Predictions already exist for {str(method)}")
        return outpath
    df = pd.read_csv(df_path)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        score = method.predict(row["sentence"])
        df.loc[i, "score"] = score
    df.to_csv(outpath, index=False)
    return outpath

def analyze(df_path):
    df = pd.read_csv(df_path)
    precision, recall, thresholds = precision_recall_curve(df["label"], df["score"])
    auc_score = auc(df["label"], df["score"])
    return precision, recall, thresholds, auc_score

def plot(plot_name, precision, recall, thresholds, auc_score):
    plt.plot(thresholds, recall[:-1], label="Recall @ Threshold")
    plt.plot(thresholds, precision[:-1], label="Precision @ Threshold")
    plt.xlabel("Thresholds")
    plt.ylabel("Precision and Recall")
    plt.title(f"{plot_name} Precision-Recall Curve (AUC={auc_score:.4f})")
    plt.savefig(f"{plot_name}_prk_curve.png")

def plot_baselines():
    df_path = "../dataset/intention/test_hf.csv"
    baselines = Baselines()
    for baseline_name in ["coin", "uniform", "label_prior", "constant_1", "constant_0"]:
        baselines.baseline_name = baseline_name
        outpath = predict(df_path, baselines)
        precision, recall, thresholds, auc_score = analyze(outpath)
        plot(baseline_name, precision, recall, thresholds, auc_score)

def plot_models():
    model_paths = ["models/intention_classifier"]
    df_path = "../dataset/intention/test_hf.csv"
    for model_path in model_paths:
        model = Model(model_path)
        outpath = predict(df_path, model)
        precision, recall, thresholds, auc_score = analyze(outpath)
        plot(str(model), precision, recall, thresholds, auc_score)


if __name__ == "__main__":
    plot_baselines()
    plot_models()