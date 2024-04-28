import json
from tqdm import tqdm
import pickle

import torch
from PIL import Image
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import json
from tqdm import tqdm
import os
import pickle
import time

OUTPUT_TRAINED_FOLDER = "approach_2"

# Given an expert, run all validation data with the expert
def eval_expert(expert_id):
    # Input data
    gp_eval_path = "json_folder/full_val_10.json"
    with open(gp_eval_path) as json_file:
        gp_eval_data = json.load(json_file)
    num_data = len(gp_eval_data)
    assert num_data == 5046

    # Load model
    disable_torch_init()
    model_path = f'/home/ubuntu/workspace/MoE-LLaVA/output_trained/{OUTPUT_TRAINED_FOLDER}/MoE-LLaVA-StableLM-1.6B-4e-okvqa_gp{expert_id}'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)

    def moeLLavaInf(image, inp):
        conv_mode = "stablelm"  # phi or qwen or stablelm
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        image_processor = processor['image']
        image_tensor = image_processor.preprocess(Image.open(image).convert('RGB'), return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

        #print(f"{roles[1]}: {inp}")
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        return outputs
    
    # Evaluate
    questionId_to_answer = {}
    save_file_id = 0
    for i in tqdm(range(num_data)):
        entry = gp_eval_data[i]
        questionId = entry['id']
        image_path = os.path.join("image_folder", entry['image'])
        input = entry['conversations'][0]['value'][8:]  # Strip "<image>\n"
        input = input
        output = moeLLavaInf(image_path, input)
        questionId_to_answer[questionId] = output
        # Checkpoint output
        if (i % 100 == 0 and i != 0) or i == num_data - 1:
            with open(f"okvqa_eval/approach2_full_eval/gp{expert_id}_outputs/saved_outputs_{save_file_id}.pkl", 'wb') as f:
                pickle.dump(questionId_to_answer, f)
            questionId_to_answer = {}
            save_file_id += 1

    # Delete to prepare for the next gp
    del tokenizer
    del model
    del processor

    time.sleep(10)


eval_expert(0)
eval_expert(1)
eval_expert(2)
