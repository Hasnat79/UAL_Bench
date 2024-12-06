#!/bin/bash
#sbatch --get-user-env=L                #replicate login env

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=hprc_job       #Set the job name to "JobExample4"
#SBATCH --time=1-1:00:00              #Set the wall clock limit to 1 day 1 hour
#SBATCH --nodes=1                #Request 1 node / 1 pc
#SBATCH --mem=32G                     #Request 32 per node | 1 pc with 32GB ram
#SBATCH --output=hprc_job_Out.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:a100:1          #Request 1 a100 GPU per node. 
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=123456789           #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=example.email@tamu.edu    #Send all emails to email_address 

# cd  /path/to/your/python/file/dir
# python [ path to your python file ] 

cd  src
# blip2 llama3
python llama3_x_blip2_text_rep_funqa.py --input ../outputs/text_representations/blip2_text_rep_x_funqa.json --output ../outputs/vlm_llm_prediction_generations/blip2_llama3_funqa.json
python llama3_x_blip2_text_rep_charades_sta.py --input ../outputs/text_representations/blip2_text_rep_x_charades.json --output ../outputs/vlm_llm_prediction_generations/blip2_llama3_charades.json
python llama3_x_blip2_text_rep_ssbd.py --input ../outputs/text_representations/blip2_text_rep_x_ssbd.json --output ../outputs/vlm_llm_prediction_generations/blip2_llama3_ssbd.json
python llama3_x_blip2_text_rep_uag_oops.py --input ../outputs/text_representations/blip2_text_rep_x_uag_oops.json --output ../outputs/vlm_llm_prediction_generations/blip2_llama3_uag_oops.json

# videollama2 llama3
python llama3_x_videollama2_text_rep_charades_sta.py --input ../outputs/text_representations/videollama2_text_rep_x_charades.json --output ../outputs/vlm_llm_prediction_generations/videollama2_llama3_charades.json
python llama3_x_videollama2_text_rep_funqa.py --input ../outputs/text_representations/videollama2_text_rep_x_funqa.json --output ../outputs/vlm_llm_prediction_generations/videollama2_llama3_funqa.json
python llama3_x_videollama2_text_rep_ssbd.py --input ../outputs/text_representations/videollama2_text_rep_x_ssbd.json --output ../outputs/vlm_llm_prediction_generations/videollama2_llama3_ssbd.json
python llama3_x_videollama2_text_rep_uag_oops.py --input ../outputs/text_representations/videollama2_text_rep_x_uag_oops.json --output ../outputs/vlm_llm_prediction_generations/videollama2_llama3_uag_oops.json

# eval 
cd src/eval
python eval_your_results.py --results_file ../../outputs/vlm_llm_prediction_generations/blip2_llama3_uag_oops.json
