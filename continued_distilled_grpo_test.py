
import torch
import os, glob
import re
from datasets import load_dataset, Dataset
from vllm import SamplingParams
import pandas as pd
import matplotlib.pyplot as plt


from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower
MODEL_NAME = "unsloth/DeepScaleR-1.5B-Preview"
ACC=[]


def get_accuracy_and_outputs(cp_folder, load_lora=True):
    torch.cuda.empty_cache()
    model, tokenizer = FastLanguageModel.from_pretrained(
        cp_folder, # model_name = MODEL_NAME,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6, # Reduce if out of memory
    )
    if load_lora: model = model.merge_and_unload()
    """### Data Prep
    <a name="Data"></a>

    We directly leverage [@willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) for data prep and all reward functions. You are free to create your own!
    """

    # Load and prep dataset
    SYSTEM_PROMPT = """
    Use as little tokens as possible to reason and come to a conclusion. You MUST respond in the following format:
    <think>
    YOUR THOUGHTS HERE
    </think>
    ...
    <think>
    YOUR THOUGHTS HERE
    </think>

    <answer>
    YOUR NUMERIC ANSWER HERE
    </answer>

    replacing YOUR THOUGHTS HERE with your thoughts and YOUR NUMERIC ANSWER HERE with your numeric answer. 
    DO NOT use the \\boxed{} format and DO NOT use the **Answer** format
    """


    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    # uncomment middle messages for 1-shot prompting
    def get_gsm8k_questions(split = "train") -> Dataset:
        data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
        data = data.map(lambda x: { # type: ignore
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        }) # type: ignore
        return data # type: ignore

    dataset = get_gsm8k_questions(split="test")[:10]



    def correctness_judge(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion.outputs[0].text for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [True if r == a else False for r, a in zip(extracted_responses, answer)]



    sampling_params = SamplingParams(
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = max_seq_length,
    )

    text = tokenizer.apply_chat_template(
        dataset['prompt'], tokenize = False, add_generation_prompt = True)

    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
    )

    corrects = correctness_judge(dataset['prompt'], output, dataset['answer'])
    accuracy = sum(corrects) / len(dataset)
    return accuracy, list(zip(dataset['prompt'], dataset['answer'], corrects))

experiment_folder = glob.glob(f"../outputs/{MODEL_NAME}*")[-1] # Get logs from most recenet files

cp_folders = glob.glob(f"{experiment_folder}/*")
# load base_model_first
accuracy, results = get_accuracy_and_outputs(cp_folders[0], load_lora=False)
ACC.append(accuracy)
res = pd.DataFrame(results)
res.to_csv(f'{cp_folders[0]}/test_trajectories.csv', index=True)  

for cp_folder in cp_folders:
    accuracy, results = get_accuracy_and_outputs(cp_folder, load_lora=True)
    print(f"Checkpoint: {cp_folder} - accuracy: {accuracy}")
    ACC.append(accuracy)
    res = pd.DataFrame(results)
    res.to_csv(f'{cp_folders[0]}/test_trajectories.csv', index=True)  # Write trajectories to csv files

import matplotlib.pyplot as plt
plt.plot(["Baseline"] + [cp.split("-")[-1]+" steps" for cp in cp_folders], ACC)
plt.title("Accuracy")
plt.savefig(f"{experiment_folder}/results.png")

