rem # generate LLM responses. input: HumanEval.jsonl, HumanEvalComm.jsonl  output: file in folder log/
python generate_response.py -d HumanEvalComm -m gpt-3.5-turbo-0125 -n 1 -t 1 -s 0 -o manualRemove -maxp -1 -s 0 -so 1
python generate_response.py -d HumanEvalComm -m gpt-3.5-turbo-0125 -n 1 -t 1 -s 0 -o manualRemove -maxp -1 -s 1 -so 2
python generate_response.py -d HumanEvalComm -m gpt-3.5-turbo-0125 -n 1 -t 1 -s 0 -o manualRemove -maxp -1 -s 2 -so 3

rem # extract code and run test cases and other metrics for each problem. input: file in log/  output: file in log/record/
python intermedia_analyze.py -f log/manualRemove_dataset_HumanEvalComm_model_gpt-3.5-turbo-0125_topn_1_temperature_1.0.log_3 -n 1

rem # compute more metrics for each problem, such as test pass rate, question quality rate, comm. rate, etc. input: file in ./log/record/ output: file in ./result_data/
python syntactic_similarity_OER.py -e manualRemove_dataset_HumanEvalComm -m gpt-3.5-turbo-0125 -t 1 -o R1 -n 1

rem # aggregate and display metrics for all problems
python measurement_summary_draw_heatmap.py -e manualRemove -d HumanEvalComm -m gpt-3.5-turbo-0125 -t 1 -n 1
