#span eval 
ITERs=(0)
for ITER in "${ITERs[@]}"
do
    task_name=rams2rams
    model_name=llama_seed_42
    data_path=dataset/${task_name}
    output_span_file=experiments/${task_name}/output_${model_name}/extractor/iter${ITER}/span_output.jsonl

    python convert_gen_to_output.py \
        --gen-file experiments/${task_name}/output_${model_name}/extractor/iter${ITER}/pred_out_single_is_eval_True_limit5000.jsonl \
        --ontology ${data_path}/ontology.csv \
        --test_file ${data_path}/test.jsonl \
        --output-file ${output_span_file} \

    python scorer/scorer.py -g=${data_path}//test.jsonl  -p=experiments/${task_name}/output_${model_name}/extractor/iter${ITER}/span_output.jsonl --task ${task_name} --unseen_role ${data_path}/unseen_role_${task_name}  \
    --reuse_gold_format --do_all > experiments/${task_name}/output_${model_name}/extractor/iter${ITER}/span_metrics.txt 
done

