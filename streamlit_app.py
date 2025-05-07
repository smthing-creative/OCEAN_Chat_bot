from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertForSequenceClassification
import torch
from copy import deepcopy
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

#chat AI model
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
#parsing both models through the GPU instead of CPU because I never tested if that would help the speed of the conversation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#OCEAN personality model
ocean_model = "Minej/bert-base-personality"
ocean_tokenizer = BertTokenizer.from_pretrained(ocean_model)
ocean_model = BertForSequenceClassification.from_pretrained(ocean_model)
ocean_model.to(device)

#chat history
chat_history = []

#OCEAN function
def ocean_traits(chat_history):
    recent_text = " ".join([msg["content"] for msg in chat_history[-6:] if msg["role"] == "user"])
    inputs = ocean_tokenizer(recent_text, truncation=True, padding=True, return_tensors="pt").to(device)
    outputs = ocean_model(**inputs)
    predictions = outputs.logits.squeeze().detach().cpu().numpy()
    trait_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    return {trait_names[i]: predictions[i] for i in range(len(trait_names))}

#adjusted system prompt to include specifically no emojis because apparently that is an issue?
def ocean_system_prompt(personality):
    return f"""Answer the questions the user has without using emojis or theatrical phrasing, include links to the sources you used. Adapt your responses to match the user's personality traits:
- Openness: {personality['openness']*100:.0f}%
- Conscientiousness: {personality['conscientiousness']*100:.0f}%
- Extraversion: {personality['extraversion']*100:.0f}%
- Agreeableness: {personality['agreeableness']*100:.0f}%
- Neuroticism: {personality['neuroticism']*100:.0f}%
Modify your tone, emotional awareness, and level of detail based on these traits.
"""

base_personality = {
    "openness": 0.5,
    "conscientiousness": 0.9,
    "extraversion": 0.2,
    "agreeableness": 0.4,
    "neuroticism": 0.1
}

def chat_ai(user_input, system_prompt):
    try:
        messages = [
            {"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1000,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        full_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return full_output
    except Exception as e:
        print("Error during generation:", e)
        return "Error occurred."

base_personality = ocean_traits(chat_history)
system_prompt = ocean_system_prompt(base_personality)
personality_log = []
message_count = 0

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        print("Exiting the chat.")
        break

    chat_history.append({"role": "user", "content": user_input})
    message_count += 1

    # Recalculating after every message and storing in a different variable than the base personality
    personality = ocean_traits(chat_history)
    print("\n[Current Personality Estimate:]")
    for trait, score in personality.items():
        print(f"  {trait.capitalize()}: {score:.2f}")

    system_prompt = ocean_system_prompt(personality)
        
    personality_log.append({
    "timestamp": datetime.now().isoformat(),
    "message_count": message_count,
    "traits": deepcopy(personality)
    })
    
    response = chat_ai(user_input, system_prompt)
    chat_history.append({"role": "assistant", "content": response})
    #print call for user_input
    print("You:", user_input)
    print("AI:", response)

    #create df for data analysis 
    df = pd.DataFrame(personality_log)
rows = []

for entry in personality_log:
    flatten = {
        "timestamp": entry["timestamp"],
        "message_count": entry["message_count"],
        **entry["traits"]
    }
    rows.append(flatten)

results = pd.DataFrame(rows)
#going to keep this in because I expect the same issue as with the test
results['timestamp'] = pd.to_datetime(results['timestamp'])
