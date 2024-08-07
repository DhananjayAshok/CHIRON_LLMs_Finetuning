from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt
import torch

universal_val_path = "../dataset/intention/validation_m.csv"
universal_test_path = "../dataset/intention/test_m.csv"

def get_outpath(df_path, method):
    df_folders = df_path.split("/")
    df_name = df_folders[-1]
    df_name = df_name.replace(".csv", f"{str(method)}_predicted.csv")
    outpath = "/".join(df_folders[:-1]) + "/predicted/" + df_name
    return outpath

class Model:
    def __init__(self, model_path, model_name="intention_clf"):
        self.model_path = model_path
        self.model_name = model_name
    
    def _init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, device_map="auto")

    def predict(self, inp):
        inputs = self.tokenizer(inp, return_tensors="pt")
        logits = self.model(**inputs).logits
        logits = logits.detach().numpy()
        softmaxed = np.exp(logits) / np.sum(np.exp(logits))
        return softmaxed[0, 1]
    
    def __str__(self):
        return self.model_name
    
    def __repr__(self):
        return str(self)


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
    outpath = get_outpath(df_path, method)
    if os.path.exists(outpath):
        print(f"Predictions already exist for {str(method)}")
        return outpath
    if isinstance(method, Model):
        method._init_model()
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
    plt.legend()
    plt.title(f"{plot_name} Precision-Recall Curve (AUC={auc_score:.4f})")
    plt.savefig(f"plots/{plot_name}_prk_curve.png")
    plt.clf()

def plot_prc(plot_name, precision, recall, auc_score):
    plt.plot(recall[:-1], precision[:-1], label=f"{plot_name} (AUC={auc_score:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve")

def plot_baselines(df_path):
    split_name = "test" if "test" in df_path else "validation"
    baselines = Baselines()
    for baseline_name in ["coin", "uniform"]:
        baselines.baseline_name = baseline_name
        outpath = predict(df_path, baselines)
        precision, recall, thresholds, auc_score = analyze(outpath)
        plot(split_name+"_"+baseline_name, precision, recall, thresholds, auc_score)

def plot_models(df_path):
    split_name = "test" if "test" in df_path else "validation"
    llama_namepaths = [("llama3_m", "models/llama3_m"), ("llama3_ms", "models/llama3_ms"), ("llama3_msc", "models/llama3_msc")]
    roberta_namepaths = [("roberta_m", "models/roberta_m"), ("roberta_ms", "models/roberta_ms"), ("roberta_msc", "models/roberta_msc")]
    model_namepaths = llama_namepaths + roberta_namepaths
    for model_name, model_path in model_namepaths:
        model = Model(model_path, model_name=model_name)
        outpath = predict(df_path, model)
        precision, recall, thresholds, auc_score = analyze(outpath)
        plot(split_name+"_"+str(model), precision, recall, thresholds, auc_score)

def do_plot_prc(df_path):
    split_name = "test" if "test" in df_path else "validation"
    baselines = Baselines()
    for baseline_name in ["coin", "uniform"]:
        baselines.baseline_name = baseline_name
        outpath = predict(df_path, baselines)
        precision, recall, thresholds, auc_score = analyze(outpath)
        plot_prc(baseline_name, precision, recall, auc_score)

    model_paths = ["models/intention_classifier"]
    for model_path in model_paths:
        model = Model(model_path)
        outpath = predict(df_path, model)
        precision, recall, thresholds, auc_score = analyze(outpath)
        plot_prc(str(model), precision, recall, auc_score)

    plt.legend()
    plt.title(f"Precision-Recall Curve")
    plt.savefig(f"plots/{split_name}_pr_curve.png")
    plt.clf()


if __name__ == "__main__":
    for df_path in [universal_val_path, universal_test_path]:
        plot_baselines(df_path)
        plot_models(df_path)
        do_plot_prc(df_path)