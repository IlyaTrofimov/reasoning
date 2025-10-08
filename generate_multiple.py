#!/usr/bin/env python
# coding: utf-8
import sys
import os
import subprocess
from os import listdir
from os.path import isfile, join
import pickle
import gc
import time
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from matplotlib import pyplot as plt
from tqdm import trange
import seaborn as sns
from gph import ripser_parallel 
from human_eval.data import write_jsonl, read_problems
from datasets import load_dataset

NUM_RETURN_SEQ = 1
CUDA = 'cuda:1'
N_LAYERS = 36
N_HEADS = 32

#model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model = "Qwen/Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token_id = tokenizer.eos_token_id

#
#
#
language_model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map=CUDA,
        torch_dtype=torch.float16,
        #output_hidden_states=True,
        output_attentions = True,
        #output_logits  = True,
        return_dict_in_generate=True
)

def generate_one_completion(task_id, prompt):

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    token_inputs = tokenizer(text, padding="longest", return_tensors="pt")

    print(task_id, token_inputs['input_ids'].shape, len(prompt))
    
    outputs = language_model.generate(input_ids = token_inputs['input_ids'].to(CUDA),
                                      attention_mask = token_inputs['attention_mask'].to(CUDA),
                                      do_sample=True, top_p = 0.95,
                                      temperature=0.8, num_return_sequences=NUM_RETURN_SEQ, max_new_tokens = 1024,
                                      output_logits = True, 
                                      output_attentions=True,
                                      return_dict_in_generate=True)

    print('OUTPUTS', outputs.sequences.shape)
    return text, token_inputs, outputs

def get_probs(prompt_len, outputs):
    probs = np.ones((NUM_RETURN_SEQ, outputs.sequences.shape[-1]))
    
    for pos in range(len(outputs.logits)):
        sm = torch.nn.Softmax(dim = 1)(outputs.logits[pos])

        for seq in range(outputs.sequences.shape[0]):
            token = outputs.sequences[seq][prompt_len + pos]
            probs[seq, prompt_len + pos] = sm[seq][token].item()

    return probs

def seq2text(model_inputs, outputs_sequences):

    # parsing
    prompt_len = len(model_inputs.input_ids[0])
    output_ids = [x[prompt_len:].tolist() for x in outputs_sequences]

    # parsing thinking content

    index = []
    thinking_content = []
    content = []

    for k in range(len(output_ids)):
        try:
            # rindex finding 151668 (</think>)
            i = len(output_ids[k]) - output_ids[k][::-1].index(151668)
        except ValueError:
            i = 0

        index.append(i)
        thinking_content.append(tokenizer.decode(output_ids[k][:i], skip_special_tokens=True).strip("\n"))
        content.append(tokenizer.decode(output_ids[k][i:], skip_special_tokens=True).strip("\n"))

    return index, thinking_content, content

def calc_mtd_data():

    code_exit = 0

    groups = []
    groups.append(list(range(N_LAYERS // 4)))
    groups.append(list(range(N_LAYERS // 4, N_LAYERS // 2)))
    #groups.append(list(range(N_LAYERS // 2, 3 * N_LAYERS // 4)))
    #groups.append(list(range(3 * N_LAYERS // 4, N_LAYERS)))

    for g in groups:
        print('GROUP', g)
        cmd = []
        for seq in range(NUM_RETURN_SEQ):
            for layer_num in g:
                cmd.append('python calc_dgms.py %d %d' % (seq, layer_num))

        with open('tmp.sh', 'w') as f:
            f.write(' & '.join(cmd) + '; wait')

        c = subprocess.call(['bash', './tmp.sh'])
        code_exit += c

        print('finished group, code_exit = ', c)
        time.sleep(1)

    all_data = {}

    if code_exit == 0:
        for seq in range(NUM_RETURN_SEQ):
            for g in groups:
                for layer_num in g:
                    data = torch.load('/att_tmp/att%d_%d.th_res' % (seq, layer_num), weights_only=False)
                    all_data.update(data)

    return all_data

def get_correct_prefix(outputs_sequences, correct_text):
    for k in range(1, outputs_sequences.shape[0]):
        out = tokenizer.decode(outputs_sequences[0:k], skip_special_tokens=True)
    
        if len(out) >= len(correct_text):
            return k

    return outputs_sequences.shape[0]

def merge_attentions(att, correct_prefixes):

    print('ATTENTIONS')
    print(len(att))
    print(len(att[-1]))
    print(att[-1][0].shape)

    att_len = att[-1][0].shape[-1]
    print('att_len', att_len)
    prompt_len = att[0][0].shape[-1]
    N_PARTS = 6
    PART_SIZE = 36 // N_PARTS

    for seq in range(NUM_RETURN_SEQ):

        #total_att = torch.zeros((36, 32, att_len, att_len), dtype = torch.float32)

        for layer_num in range(N_LAYERS):

            total_att = torch.zeros((32, att_len, att_len), dtype = torch.float32)

            for i in range(len(att)): # generation step
                m_layers = att[i]
                m_heads = m_layers[layer_num]

                step_num = i - 1

                if i == 0:
                    #total_att[layer_num, :, :prompt_len, :prompt_len] = m_heads[seq].cpu()
                    total_att[:, :prompt_len, :prompt_len] = m_heads[seq].cpu()
                else:
                    num_elem = m_heads.shape[-1]
                    #total_att[layer_num, :, num_elem-1, :num_elem] = m_heads[seq].cpu().squeeze()
                    total_att[:, num_elem-1, :num_elem] = m_heads[seq].cpu().squeeze()

            #A = total_att[part*PART_SIZE:(part+1)*PART_SIZE, :, :correct_prefixes[seq], :correct_prefixes[seq]].clone()
            A = total_att[:, :correct_prefixes[seq], :correct_prefixes[seq]]
            print('saving %s /att_tmp/att%d_%d.th' % (str(A.shape), seq, layer_num))
            torch.save(A, '/att_tmp/att%d_%d.th' % (seq, layer_num))
            del A
            del total_att
            gc.collect()

ds = load_dataset("openai/gsm8k", "main")['test']

for task_num in range(len(ds)):
    print('QUESTION')
    print(ds[task_num]['question'])
    print('ANSWER')
    print(ds[task_num]['answer'])
    print()

    prompt = ds[task_num]['question']

    results = {}
    results_code = {}

    #MAX_TOTAL_TOKENS = 1500
    #MAX_PROGRAM = 3500
    #program = program[0:MAX_PROGRAM]

    print('RUNNING', task_num)

    for seed in range(1):
        set_seed(seed)
        print(task_num, seed)
        print('-------------PROMPT-------------')
        print(prompt)

        new_prompt, token_inputs, outputs = generate_one_completion(task_num, prompt)

        #token_inputs = tokenizer(prompt, padding="longest", return_tensors="pt")
        print("token_inputs", token_inputs['input_ids'].shape)

        #correct_prefixes = [min(outputs.sequences.shape[1] - 1, MAX_TOTAL_TOKENS)]
        index, think, answer = seq2text(token_inputs, outputs.sequences)

        for i in range(NUM_RETURN_SEQ):
            print("---- INDEX ----")
            print(index[i])
            print("---- THINKING ----")
            print(think[i]) # no opening <think> tag
            print("---- CONTENT -----")
            print(answer[i])

        correct_prefixes = [get_correct_prefix(outputs.sequences[i], new_prompt + think[i] + answer[i]) for i in range(NUM_RETURN_SEQ)]
        print('correct_prefixes', correct_prefixes)
        #continue
        os.system('rm -rf /att_tmp/*')
        merge_attentions(outputs.attentions, correct_prefixes)
        print('merged_att DONE')

        prompt_len = token_inputs['input_ids'].shape[1]
        probs = get_probs(prompt_len, outputs)
        print('get_probs DONE')

        results_code[(task_num, seed)] = ((think, answer), outputs.sequences.cpu().clone(), probs)

        del outputs
        torch.cuda.empty_cache()

        print('calc_dgms START')
        print(prompt_len, index)
        torch.save((prompt_len, index), '/att_tmp/prompt_len.th')
        par_data = calc_mtd_data()
        print('calc_dgms DONE')
        
        continue

        for seq in range(NUM_RETURN_SEQ):
            for layer in range(N_LAYERS // 2):
                for head in range(N_HEADS):

                    diag_sum1, diag_sum2, mtd1, mtd2 = par_data[(seq, layer, head)]

                    data = [prompt_len, correct_prefixes[seq], diag_sum1, diag_sum2]
                    data.extend([mtd1, mtd2])

                    # seq replaced by num
                    results[(task_num, seed, seq, layer, head)] = data

    # NUM IS NOT DEFINED
    pickle.dump(results, open('./gsm8k/%d.pickle' % task_num, 'wb'))
    pickle.dump(results_code, open('./gsm8k/%d_code.pickle' % task_num, 'wb'))

    gc.collect()
