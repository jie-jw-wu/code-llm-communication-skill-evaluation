
python generate_response.py -d HumanEvalComm -m Okanagan -n 1 -t 1 -s 0 -o manualRemove -maxp -1 -s 0 -so 1
python generate_response.py -d HumanEvalComm -m Okanagan -n 1 -t 1 -s 0 -o manualRemove -maxp -1 -s 1 -so 2
python generate_response.py -d HumanEvalComm -m Okanagan -n 1 -t 1 -s 0 -o manualRemove -maxp -1 -s 2 -so 3

python intermedia_analyze.py -f log/manualRemove_dataset_HumanEvalComm_model_Okanagan_topn_1_temperature_1.0.log_3 -n 1

python syntactic_similarity_OER.py -e manualRemove_dataset_HumanEvalComm -m Okanagan -t 1 -o R1 -n 1

python measurement_summary_draw_heatmap.py -e manualRemove -d HumanEvalComm -m Okanagan -t 1 -n 1
