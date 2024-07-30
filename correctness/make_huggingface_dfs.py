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
    columns = ["message", "speaker", "receiver", "sender_label", "receiver_label", "game_score_delta", "time_code", "cicero_orders", "game_id","absolute_message_index","relative_message_index"]
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
            data.append([row["messages"][j], row["speakers"][j], row["receivers"][j], row["sender_labels"][j], row["receiver_labels"][j], eval(row['game_score_delta'])[j], timecode, cicero_orders, row["game_id"], row['absolute_message_index'], row["relative_message_index"]])
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

def construct_hf_data(df):
    df["sentence"] = df["speaker"] + "has score difference of" + df["game_score_delta"].astype(str)+ " and " + df["cicero_orders"] + " sends " + df["receiver"] + ": "+df["message"]
    df["label"] = (df["sender_label"] == False) | (df["receiver_label"] == False)
    return df[['sentence', 'label']]

if __name__ == "__main__":
    print(f"This script should be run from the correctness folder (not root) or it will cause bugs")
    train, test, val = get_dataset()
    train = construct_hf_data(train)
    test = construct_hf_data(test)
    val = construct_hf_data(val)
    train.to_csv('../dataset/intention/train_hf.csv', index=False)
    test.to_csv('../dataset/intention/test_hf.csv', index=False)
    val.to_csv('../dataset/intention/validation_hf.csv', index=False)

