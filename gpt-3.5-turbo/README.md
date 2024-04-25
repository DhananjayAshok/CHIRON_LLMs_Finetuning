# CHIRON_LLMs_finetuning-Gpt Generation

## Installation
1.create a new virtual environment and install openai

```bash
pip install openai
```
2.create folders named as game_info and NEW_FILES
```bash
mkdir game_info
mkdir NEW_FILES
```
3.add your own openai API key into gpt.py

4.run meesage_filter.py to pre-process data, if you want to test on game12 then just use tes.jsonl(which contains game4 and game12, remember to remove game4)

5.run gpt.py
```bash
python message_filter.py
python gpt.py
```
