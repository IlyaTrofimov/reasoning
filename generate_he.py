#!/usr/bin/env python
# coding: utf-8
import sys
import os
import pickle
import gc
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from matplotlib import pyplot as plt
from tqdm import trange
import seaborn as sns
from gph import ripser_parallel 
from human_eval.data import write_jsonl, read_problems

NUM_RETURN_SEQ = 5
CUDA = 'cuda:0'

model = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token_id = tokenizer.eos_token_id

language_model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map=CUDA,
        torch_dtype=torch.float16,
        #output_hidden_states=True,
        output_attentions = True,
        #output_logits  = True,
        return_dict_in_generate=True
).eval()


stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]

def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

def get_correct_prefix(outputs_sequences, correct_text):
    for k in range(1, outputs_sequences.shape[0]):
        out = tokenizer.decode(outputs_sequences[0:k], skip_special_tokens=True)
    
        if len(out) >= len(correct_text):
            return k

    return outputs_sequences.shape[0]


def generate_one_completion(task_id, prompt):
    token_inputs = tokenizer(prompt, padding="longest", return_tensors="pt")

    print(task_id, token_inputs['input_ids'].shape, len(prompt))
    
    outputs = language_model.generate(input_ids = token_inputs['input_ids'].to(CUDA),
                                      attention_mask = token_inputs['attention_mask'].to(CUDA),
                                      do_sample=True, top_p = 0.95,
                                      temperature=0.8, num_return_sequences=5, max_length=512,
                                      output_logits = True, 
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id = tokenizer.eos_token_id)

    return outputs

def seq2text(prompt, outputs_sequences):
    res = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs_sequences]
    answer = [x[len(prompt):] for x in res]
    res_clean =  [prompt + _stop_at_stop_token(x, stop_words) for x in answer]

    return res_clean

problems = read_problems()

def get_probs(prompt_len, outputs):
    probs = np.ones((5, 512))
    
    for pos in range(len(outputs.logits)):
        sm = torch.nn.Softmax(dim = 1)(outputs.logits[pos])

        for seq in range(outputs.sequences.shape[0]):
            token = outputs.sequences[seq][prompt_len + pos]
            probs[seq, prompt_len + pos] = sm[seq][token].item()
    return probs

def get_all_data():
    cmd = []

    for seq in range(NUM_RETURN_SEQ):
        for part in range(2):
            cmd.append('python calc_dgms.py %d %d' % (seq, part))

    code_exit = os.system(' & '.join(cmd) + '; wait')
    print('finished group 1', code_exit)

    time.sleep(5)

    cmd = []

    for seq in range(NUM_RETURN_SEQ):
        for part in range(2, 4):
            cmd.append('python calc_dgms.py %d %d' % (seq, part))

    code_exit += os.system(' & '.join(cmd) + '; wait')
    print('finished group 2', code_exit)

    all_data = {}

    if code_exit == 0:
        for seq in range(NUM_RETURN_SEQ):
            for part in range(4):
                data = torch.load('/att_tmp/att%d_%d.th_res' % (seq, part), weights_only=False)
                all_data.update(data)

    return all_data

def merge_attentions(att, correct_prefixes):

    att_len = att[-1][0].shape[-1]
    print('att_len', att_len)
    prompt_len = att[0][0].shape[-1]

    for seq in range(NUM_RETURN_SEQ):

        total_att = torch.zeros((32, 32, att_len, att_len))

        for i in range(len(att)):
            m_layers = att[i]

            for j in range(len(m_layers)):
                m_heads = m_layers[j]

                step_num = i - 1
                layer_num = j

                if i == 0:
                    total_att[layer_num, :, :prompt_len, :prompt_len] = m_heads[seq].cpu()
                else:
                    num_elem = m_heads.shape[-1]
                    total_att[layer_num, :, num_elem-1, :num_elem] = m_heads[seq].cpu().squeeze()

        for part in range(4):
            A = total_att[part*8:(part+1)*8, :, :correct_prefixes[seq], :correct_prefixes[seq]].clone()
            print(A.shape)
            torch.save(A, '/att_tmp/att%d_%d.th' % (seq, part))
            del A

    gc.collect()

#
#   MAIN LOOP
#
for task_id in problems:

    results = {}
    results_code = {}

    task_num = int(task_id.split('/')[-1])

    #if task_num != 14:
    #    continue

    for seed in range(5):
        set_seed(seed)
        print(task_id, seed)
        print('-------------PROMPT-------------')
        print(problems[task_id]["prompt"])

        outputs = generate_one_completion(task_id, problems[task_id]["prompt"])
        code_clean = seq2text(prompt, outputs.sequences)
        correct_prefixes = [get_correct_prefix(outputs.sequences[i], code_clean[i]) for i in range(NUM_RETURN_SEQ)]

        print(correct_prefixes)
        os.system('rm -rf /att_tmp/*')
        merge_attentions(outputs.attentions, correct_prefixes)
        print('merged_att DONE')

        prompt_len = outputs.attentions[0][0].shape[-1]
        probs = get_probs(prompt_len, outputs)
        print('get_probs DONE')

        prompt_len_chars = len(problems[task_id]["prompt"])
        full_text = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs.sequences]

        for i, c in enumerate(code_clean):
            print('-------------MODEL ANSWER-------------')
            print('-----------------FULL-----------------')
            print(full_text[i])
            print('-----------------CLEAN-----------------')
            print(c[prompt_len_chars:])


        results_code[(task_id, seed)] = ([c[prompt_len_chars:] for c in code_clean], outputs.sequences.cpu().clone(), probs)

        del outputs
        torch.cuda.empty_cache()

        #
        #
        #
        torch.save(prompt_len, '/att_tmp/prompt_len.th')
        par_data = get_all_data()
        print('calc_dgms DONE')

        for seq in range(NUM_RETURN_SEQ):
            for layer in range(32):
                for head in range(32):

                    diag_sum1, diag_sum2, mtd1, mtd2 = par_data[(seq, layer, head)]

                    data = [prompt_len, correct_prefixes[seq], diag_sum1, diag_sum2]
                    data.extend([mtd1, mtd2])

                    results[(task_id, seed, seq, layer, head)] = data

    pickle.dump(results, open('/codellama_results2/%d.pickle' % task_num, 'wb'))
    pickle.dump(results_code, open('/codellama_results2/%d_code.pickle' % task_num, 'wb'))

    gc.collect()
