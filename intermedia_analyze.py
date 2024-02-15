import argparse
import json
import os
import openai
import re
import subprocess

import nltk
import pycode_similar
from nltk.translate.bleu_score import sentence_bleu

PROMPT_EVALUATE_QUESTIONS = 'Some information is removed from the original problem description. Questions are expected to get the information. Given the missing information, problem description, evaluate the quality of the questions. Return only an integer: 3 (Good), 2 (Fair), or 1 (Bad). ### Questions: {clarifying_questions} ### Problem Description: {problem} ### original description(includes missing information): {missing_information} \n'

# TODO(jwu): adjust prompt
def evaluate_clarifying_questions(
    missing_information='',
    clarifying_questions='',
    problem=''
):
    topn = 1
    temperature = 1.0
    model = 'gpt-3.5-turbo'
    completion = openai.ChatCompletion.create(
        model=model,
        n=topn,
        temperature=temperature,
        messages=[{
            "role": "user",
            "content": PROMPT_EVALUATE_QUESTIONS.format(
                missing_information=missing_information,
                clarifying_questions=clarifying_questions,
                problem=problem
            )
        }]
    )
    response_list = []
    for i in completion['choices']:
        response_list.append(i['message']['content'])
    # assume the result has only one element (n=1) which is only int
    return ''.join(filter(str.isdigit, response_list[0]))

# TODO(jwu): bug this code return last triple code snippet. 
def response_2_code(response):
    code_template = re.compile('```.*\n([\s\S]+?)\n```', re.M)
    code = code_template.findall(response)
    if len(code) > 0:
        return code[-1]
    else:
        return ''

# returns code only if the response consists solely of code with markups
def response_2_code_if_no_text(response):
    code_template = re.compile('^```.*\n([\s\S]+?)\n```$', re.M)
    code = code_template.findall(response)
    if len(code) > 0:
        return code[-1]
    else:
        return ''

def solution_evaluation(solution, test_cases, demo_file, time_limit):
    passed_case = []
    case_status = []
    with open(demo_file, 'w') as f:
        f.write(solution)
    for i in range(len(test_cases)):
        try:
            # TODO: timeout value survey
            output = subprocess.run(["python", demo_file], capture_output=True, text=True,
                                    input=test_cases[i]['input'], timeout=time_limit)
        except subprocess.TimeoutExpired as e:
            print(e, flush=True)
            case_status.append('timeout')
            continue
        except Exception as e:
            print(e, flush=True)
            case_status.append('exception')
            continue
        if output.returncode != 0:
            case_status.append('execution error: %s' % output.returncode)
        else:
            case_status.append(output.stdout.strip())
        if test_cases[i]['output'].strip() == output.stdout.strip():
            passed_case.append(i)

    pass_num = len(passed_case)
    print('%s/%s pass.' % (pass_num, len(test_cases)), flush=True)
    return passed_case, case_status

def solution_evaluation_HumanEval(solution, test_cases, demo_file, call_demo_file, entry_point, time_limit):
    passed_case = []
    case_status = []
    with open(demo_file, 'w', encoding='utf-8') as f:
        f.write(solution)
    for i in range(len(test_cases)):
        if test_cases[i]['relation'] == '==':
            with open(call_demo_file, 'w') as f:
                f.write('from %s import %s\nprint(%s(%s))' % (
                    demo_file.split('.')[0],
                    entry_point,
                    entry_point,
                    test_cases[i]['input']
                ))
            try:
                output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=time_limit)

            except subprocess.TimeoutExpired as e:
                print(e, flush=True)
                case_status.append('Timeout')
                continue
            except Exception as e:
                print(e, flush=True)
                case_status.append('Exception')
                continue
            if output.returncode != 0:
                case_status.append('execution error: %s' % output.returncode)
            else:
                case_status.append(output.stdout.strip())
            if test_cases[i]['output'].strip() == output.stdout.strip():
                passed_case.append(i)
        else:
            if '$input$' in test_cases[i]['relation'] or '$demo$' in test_cases[i]['relation']:
                with open(call_demo_file, 'w') as f:
                    f.write('from %s import %s\n%s' % (
                        demo_file.split('.')[0],
                        entry_point,
                        test_cases[i]['relation'].replace('$input$', str(test_cases[i]['input'])).replace('$demo$', demo_file.split('.')[0])
                    ))
            else:
                with open(call_demo_file, 'w') as f:
                    f.write('from %s import %s\nprint(%s)' % (demo_file.split('.')[0],
                        entry_point,
                        test_cases[i]['relation'].replace('candidate', entry_point)))
                try:
                    output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=time_limit)

                except subprocess.TimeoutExpired as e:
                    print(e, flush=True)
                    case_status.append('Timeout')
                    continue
                except Exception as e:
                    print(e, flush=True)
                    case_status.append('Exception')
                    continue
                if output.returncode != 0:
                    case_status.append('execution error: %s' % output.returncode)
                else:
                    case_status.append(output.stdout.strip())
                if output.stdout.strip() == 'True':
                    passed_case.append(i)

    pass_num = len(passed_case)
    print('%s/%s pass.' % (pass_num, len(test_cases)), flush=True)
    return passed_case, case_status

def analyze_process_HumanEval(log_file, original_prompt_file, topn):
    demo_file = 'demo.py'
    call_demo_file = 'call_demo.py'
    count = 0
    problem_dic = {}
    while os.path.exists(demo_file) or os.path.exists(call_demo_file):
        demo_file = 'demo_%s.py' % count
        call_demo_file = 'call_demo_%s.py' % count
        count += 1
    names = []
    if not os.path.exists('./log/record/%s' % (log_file.split('/')[1])):
        with open('./log/record/%s' % (log_file.split('/')[1]), 'w') as f:
            f.write('')
    else:
        with open('./log/record/%s' % (log_file.split('/')[1]), 'r') as f:
            for line in f.readlines():
                content = json.loads(line)
                names.append(content['name'])
    problem_list = []
    # TODO(jwu): we can read from HumanEval.jsonl instead of HumanEval_new.jsonl, and delete all about HumanEval_new 
    # with open('HumanEval/HumanEval.jsonl', 'r') as f:
    with open('HumanEval/HumanEval_new.jsonl', 'r') as f:
        for line in f.readlines():
            problem_list.append(json.loads(line))
            # added by JW. not needed since it's just loading HumanEval problems.
            #break

    for i in range(len(problem_list)):
        if not problem_list[i]['name'] in names:
            problem_dic[problem_list[i]['name']] = {
                'name': problem_list[i]['name'],
                'index_num': i,
                'time_limit': int(3) # by default
            }

    # get results from original prompts (without random removing content in prompts)
    original_prompt_dic = {}
    original_prompt_case_status_dic = {}
    if original_prompt_file != '':
        with open(original_prompt_file, 'r') as f:
            for line in f.readlines():
                content = json.loads(line)
                name = content['name']
                original_prompt_dic[name] = content['code_candidates']

    #JW: for each response in log, construct `problem_dic`

    with open(log_file, 'r') as f:
        for line in f.readlines():
            content = json.loads(line)
            name = content['name']
            if name in names:
                continue
            index = content['index']
            response = content['response']
            original_prompt = content['original_prompt']
            modified_prompt = content['modified_prompt']
            if index == 0:
                print('----------------------problem name: %s--------------------------------' % (name),
                      flush=True)
            # initialize
            if 'code_candidates' not in problem_dic[name]:
                problem_dic[name]['response_candidates'] = []
                problem_dic[name]['code_candidates'] = []
            print('generate code from response', flush=True)
            # load from code_contest dataset
            problem = problem_list[problem_dic[name]['index_num']]
            test_set = problem['test_case']
            reference_code = []
            reference_code.append(problem['solution'])

            # get code from response
            code = response_2_code(response)
            # default weight: weights=(0.25, 0.25, 0.25, 0.25)
            # if reference_code == []:
            #     BLEU_score_correct = -1
            # else:
            #     BLEU_score_correct = sentence_bleu(reference_code, code.split())

            # use code to run test cases
            time_limit = problem_dic[name]['time_limit']
            question_quality_result = '0'
            if original_prompt_file != '' and code == '':
                # response is asking questions. communication success. use original prompt results in this case
                # TODO(jwu): we should continue to provide answers to the quetions, and ask to generate code again. Then compute test pass rate.
                test_case_solved = [original_prompt_dic[name][index]['passed_case'], original_prompt_dic[name][index]['case_status']]
                # evaluate clarifying questions
                question_quality_result = evaluate_clarifying_questions(original_prompt,response,modified_prompt)
            else:
                test_case_solved = solution_evaluation_HumanEval(code, test_set, demo_file, call_demo_file, problem['entry_point'], time_limit)
            res = {
                'code': code,
                'index': index,
                'passed_case': test_case_solved[0],
                'case_status': test_case_solved[1],
                'question_quality': question_quality_result, 
                # 'BlEU_score_correct': BLEU_score_correct
            }
            problem_dic[name]['response_candidates'].append(response)
            problem_dic[name]['code_candidates'].append(res)
            if index == topn - 1:
                print('%s stability analyze' % (name), flush=True)
                print('writing in %s' % (name), flush=True)
                # write in
                json_str = json.dumps(problem_dic[name])
                with open('./log/record/%s' % (log_file.split('/')[1]), 'a') as f:
                    f.write(json_str + '\n')
                problem_dic.pop(name)

# extract code and run test cases
# input: file in log/
# output: file in log/record/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Choose file",
        required=True,
    )

    # file of results with original prompt
    parser.add_argument(
        "-of",
        "--originalFile",
        type=str,
        help="Choose file",
        default="",
    )

    parser.add_argument(
        "-n",
        "--topn",
        type=int,
        help="Top N candidates",
        default=5,
    )

    args = parser.parse_args()
    if 'HumanEval' in args.file:
        analyze_process_HumanEval(args.file, args.originalFile, args.topn)
