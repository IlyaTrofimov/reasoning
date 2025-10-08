import os
from code_eval import compute_code_eval
from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "0"

def get_prompt(doc):
    """Builds the prompt for the LM to generate from.
    MBPP prompt is built following to InCoder (Fried et al.) approach
    prompt = docstring that includes one test
    """
    description = doc["text"]
    test_example = doc["test_list"][0]
    prompt = f'"""\n{description}\n{test_example}\n"""\n'
    return prompt

def get_reference(doc):
    """Builds the reference solution for the doc (sample from the test dataset)."""
    return "\n".join(doc["test_list"])

problems = {}

cnt = 0
for idm, elem in enumerate(stream_jsonl('/mbpp/mbpp.jsonl')):
    k = 'MBPP/%d' % cnt

    problems[k] = {'task_id' : k, 'prompt' : get_prompt(elem), 'test_list' : elem['test_list']}
    cnt+=1

correct = 0
cnt = 0

all_results = []

for elem in stream_jsonl('mbpp_samples.jsonl'):
    #print(elem.keys())
    #print()
    #print(get_prompt(elem))

    #print('elem')
    #print(elem)

    references = get_reference(problems[elem['task_id']])

    #print('referenes')
    #print(references)

    results1, results2 = compute_code_eval(
                references=[references],
                predictions=[elem['completions']])

    #print(results1)
    all_results.append({'task_id' : elem['task_id'], 'results' :  results2[0]})

    correct += results1['pass@1']
    cnt += 1
    #print(references)

print(correct, cnt, correct / cnt)

write_jsonl("mbpp_evaluation.jsonl", all_results)
