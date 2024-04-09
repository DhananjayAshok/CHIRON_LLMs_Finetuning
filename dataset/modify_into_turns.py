import os
import json

if __name__ == "__main__":
    def main():
        folder_path = "human_game/Training"
        file_names = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        country_mapping = {'AUS':'AUSTRIA','ENG':'ENGLAND','FRA':'FRANCE','ITA':'ITALY','RUS':'RUSSIA','GER':'GERMANY','TUR':'TURKEY'}
        for file_name in file_names:
            new_data =[]
            print(file_name)
            power_user = file_name.split('_')[2]
            power_counterpart = file_name.split('_')[3].split('.')[0]
            with open(f'{folder_path}/{file_name}', 'r') as file:
                data = json.load(file)
            for phases in data:
                new_phases = phases
                messages_turns = []
                dict_temp = {'message_send':'','message_counterpart':'','message_suggestion':''}
                messages = phases.get('messages', [])
                for message in messages:
                    if message["sender"] == country_mapping[power_user]:
                        dict_temp['message_send'] = dict_temp['message_send']+message["input"]+' '
                    if message["sender"] == country_mapping[power_counterpart]:
                        dict_temp['message_counterpart'] = message["input"]
                        dict_temp['message_suggestion'] = message["output"]
                        messages_turns.append(dict_temp)
                        dict_temp = {'message_send':'','message_counterpart':'','message_suggestion':''}
                new_phases['messages'] = messages_turns
                new_data.append(new_phases)
            output_file_path = f"human_game/New_Training/{file_name}"
            with open(output_file_path, 'w') as output_file:
                json.dump(new_data, output_file, indent=2)
            print(f"whole_phases processed and saved to {output_file_path}")
    main()

