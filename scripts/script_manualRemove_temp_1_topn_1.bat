rem Example: 
rem XXX.bat 0 165 gpt-3.5-turbo-0125
rem XXX.bat 0 165 Okanagan
python generate_response.py -d HumanEvalComm -m %3 -n 1 -t 1 -o manualRemove -minp %1 -maxp %2 --log_phase_input 0 --log_phase_output 1
python generate_response.py -d HumanEvalComm -m %3 -n 1 -t 1 -o manualRemove -minp %1 -maxp %2 --log_phase_input 1 --log_phase_output 2
python generate_response.py -d HumanEvalComm -m %3 -n 1 -t 1 -o manualRemove -minp %1 -maxp %2 --log_phase_input 2 --log_phase_output 3
