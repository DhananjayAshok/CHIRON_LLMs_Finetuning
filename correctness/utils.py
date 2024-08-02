from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt
import torch

def predict(df_path, model_path):
    df = pd.read_csv(df_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        inputs = tokenizer(row["sentence"], return_tensors="pt")
        logits = model(**inputs).logits
        softmax = torch.nn.functional.softmax(logits, dim=1)
        df.loc[i, "score"] = softmax[0, 1]
    outpath = df_path.replace(".csv", "_predicted.csv")
    return outpath

def analyze(df_path):
    df = pd.read_csv(df_path)
    precision, recall, thresholds = precision_recall_curve(df["label"], df["score"])
    auc_score = auc(df["label"], df["score"])
    return precision, recall, thresholds, auc_score

def plot(precision, recall, thresholds, auc_score):
    plt.plot(thresholds, recall, label="Recall @ Threshold")
    plt.plot(thresholds, precision, label="Precision @ Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Precision and Recall")
    plt.title(f"Precision-Recall Curve (AUC={auc_score:.2f})")
    plt.savefig("prk_curve.png")


if __name__ == "__main__":
    df_path = "../dataset/intention/test_hf.csv"
    model_path = "model/intention_classifier"
    outpath = predict(df_path, model_path)
    precision, recall, thresholds, auc_score = analyze(outpath)
    plot(precision, recall, thresholds, auc_score)