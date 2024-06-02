import subprocess

# https://docs.coqui.ai/en/latest/models/xtts.html

env = "pipenv run"
model_name = '--model_name tts_models/multilingual/multi-dataset/xtts_v2'
language_idx = '--language_idx en'
speaker_idx = "--speaker_idx 'Claribel Dervla'"

with open ('data/input.txt', 'r') as f:
    text = f'--text "{f.read()[:100]}"'
# text = "--text 'Hello World'"

out_path = 'data/output.wav'
output_path = f"--out_path {out_path}"

cmd = f"tts {model_name} {language_idx} {speaker_idx} {output_path} {text}"
command = f"{env} {cmd}"

print(f'command {command}\n')

try:
    kwargs = { "shell": True, "capture_output": True, "text": True }
    result = subprocess.run(command, **kwargs)
    print(f'result {result}\n')
    print(f'output {result.stdout}\n')

    result = subprocess.run(f"afplay {out_path}", **kwargs)
    print(f'result {result}\n')
    print(f'output {result.stdout}\n')
except Exception as e:
    print(f'error {e}\n')

