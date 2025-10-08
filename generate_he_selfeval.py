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

NUM_RETURN_SEQ = 1
CUDA = 'cuda:2'

model = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token_id = tokenizer.eos_token_id

language_model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map=CUDA,
        torch_dtype=torch.float16,
        #output_hidden_states=True,
        #output_attentions = True,
        #output_logits  = True,
        return_dict_in_generate=True
).eval()


#stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
stop_words=["[DONE]"]

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


def generate_one_completion(prompt):
    token_inputs = tokenizer(prompt, padding="longest", return_tensors="pt")

    print(token_inputs['input_ids'].shape, len(prompt))
    
    outputs = language_model.generate(input_ids = token_inputs['input_ids'].to(CUDA),
                                      attention_mask = token_inputs['attention_mask'].to(CUDA),
                                      do_sample=False, top_p = 0.95,
                                      temperature=0.8, num_return_sequences=1, max_new_tokens = 64,
                                      output_logits = True, 
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id = tokenizer.eos_token_id)

    return outputs

def clean(s):
    idx = s.find('[DONE]')
    return s[0:idx]

def seq2text(prompt, outputs_sequences):
    res = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs_sequences]
    answer = [x[len(prompt):] for x in res]
    #res_clean =  [prompt + _stop_at_stop_token(x, stop_words) for x in answer]
    return answer

    #return answer

problems = read_problems()
#print(problems)
#pickle.dump(problems, open('/he_problems.pickle', 'wb'))
#sys.exit()

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

tasks = pickle.load(open('self_eval_he_tasks.pickle', 'rb'))

#
#   MAIN LOOP
#
for idx in trange(len(tasks)):

    prompt = tasks[idx]

    seed = 1

    set_seed(seed)
    outputs = generate_one_completion(prompt)

    code_clean = seq2text(prompt, outputs.sequences)
    print('---------------------')
    print(prompt)
    print('---------------------')
    print(code_clean)
    print('---------------------')
    del outputs
    torch.cuda.empty_cache()

    pickle.dump(code_clean, open('/self_eval_he/%d.pickle' % idx, 'wb'))

    gc.collect()
