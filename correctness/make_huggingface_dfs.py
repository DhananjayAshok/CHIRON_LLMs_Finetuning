import pandas as pd
import os
from tqdm import tqdm
from nlimetric import report_move

def proc_df(df, list_cols):
    for col in list_cols:
        df[col] = df[col].apply(eval)
    return

def expand_df(df):
    cicero_path = "../dataset/human_game/Cicero_orders_dataset/"
    data = []
    columns = ["message", "message_history", "speaker", "receiver", "sender_label", "receiver_label", "game_score_delta", "time_code", "cicero_orders", "game_id","absolute_message_index","relative_message_index"]
    for i, row in tqdm(df.iterrows(), total=len(df)):
        for j in range(len(row["messages"])):
            timecode = eval(row["seasons"])[j][0]+eval(row["years"])[j]
            game = row["game_id"]
            potential_cicero_path = cicero_path + f"humangame{game}_cicero_orders.json"
            if not os.path.exists(potential_cicero_path):
                cicero_orders = ""
            else:
                cicerodf = pd.read_json(potential_cicero_path).dropna().reset_index(drop=True)
                phase = cicerodf[cicerodf['phase'].apply(lambda x: timecode in x)].reset_index(drop=True)
                if len(phase) != 1:
                    cicero_orders = ""
                else:
                    all_orders = phase['cicero_orders'].values[0]
                    moves = None
                    for orders in all_orders:
                        if list(orders.keys())[0] == row["speakers"][j].upper():
                            moves = orders[row['speakers'][j].upper()]
                            break
                    if moves is None:
                        cicero_orders = ""
                    else:
                        cicero_orders = report_move(row["speakers"][j].upper(), moves)
            history = []
            for k in range(j):
                history.append(row["speakers"][k] + ": "+ row["messages"][k])
            data.append([row["messages"][j], history, row["speakers"][j], row["receivers"][j], row["sender_labels"][j], row["receiver_labels"][j], eval(row['game_score_delta'])[j], timecode, cicero_orders, row["game_id"], row['absolute_message_index'], row["relative_message_index"]])
    return pd.DataFrame(data, columns = columns)

def get_dataset():
    list_cols = ["messages","sender_labels", "receiver_labels", "speakers" ,"receivers"]
    train = pd.read_csv('../dataset/intention/train.csv')
    test = pd.read_csv('../dataset/intention/test.csv')
    val = pd.read_csv('../dataset/intention/validation.csv')
    for df in [train, test, val]:
        proc_df(df, list_cols)

    train = expand_df(train)
    test = expand_df(test)
    val = expand_df(val)
    return train, test, val

def construct_hf_datas(df):
    df["label"] = (df["sender_label"] == False) | (df["receiver_label"] == False).astype(int)
    df["m_text"] = df["speaker"] + " sends " + df["receiver"] + ": "+df["message"]
    df["s_text"] = "score difference: " + df["game_score_delta"].astype(str)
    df["c_text"] = df["speaker"] + "will move " + df["cicero_orders"]
    df["h_text"] = "history: " + df["message_history"].apply(lambda x: " ".join(x))
    variants = {"m": None, "ms": None, "msc": None, "msch": None}
    df["sentence"] = df["m_text"]
    variants["m"] = df[["sentence", "label"]]
    df["sentence"] = df["s_text"] + "|" + df["m_text"]
    variants["ms"] = df[["sentence", "label"]]
    df["sentence"] = df["c_text"] + "|" + df["s_text"] + "|" + df["m_text"]
    variants["msc"] = df[["sentence", "label"]]
    df["sentence"] = df["h_text"] + "|" + df["c_text"] + "|" + df["s_text"] + "|" + df["m_text"]
    variants["msch"] = df[["sentence", "label"]]
    return variants

if __name__ == "__main__":
    print(f"This script should be run from the correctness folder (not root) or it will cause bugs")
    train, test, val = get_dataset()
    trains = construct_hf_datas(train)
    tests = construct_hf_datas(test)
    vals = construct_hf_datas(val)
    base_path = "../dataset/intention/"
    for key in trains.keys():
        trains[key].to_csv(base_path + f"train_{key}.csv", index=False)
        tests[key].to_csv(base_path + f"test_{key}.csv", index=False)
        vals[key].to_csv(base_path + f"val_{key}.csv", index=False)