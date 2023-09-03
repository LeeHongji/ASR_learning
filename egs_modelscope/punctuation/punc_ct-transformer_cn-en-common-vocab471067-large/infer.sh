#!/usr/bin/env bash

set -e
set -u
set -o pipefail

stage=1
stop_stage=2
model="damo/punc_ct-transformer_cn-en-common-vocab471067-large"
model_revision="v1.0.0"
data_dir="./data"
output_dir="./results"
gpu_inference=true    # whether to perform gpu decoding
gpuid_list="0,1"    # set gpus, e.g., gpuid_list="0,1"
njob=64    # the number of jobs for CPU decoding, if gpu_inference=false, use CPU decoding, please set njob
checkpoint_dir=
checkpoint_name="punc.pb"

. utils/parse_options.sh || exit 1;

if ${gpu_inference} == "true"; then
    nj=$(echo $gpuid_list | awk -F "," '{print NF}')
else
    nj=$njob
    gpuid_list=""
    for JOB in $(seq ${nj}); do
        gpuid_list=$gpuid_list"-1,"
    done
fi

mkdir -p $output_dir/split
split_scps=""
for JOB in $(seq ${nj}); do
    split_scps="$split_scps $output_dir/split/text.$JOB.scp"
done
perl utils/split_scp.pl ${data_dir}/punc_example.txt ${split_scps}

if [ -n "${checkpoint_dir}" ]; then
  python utils/prepare_checkpoint.py ${model} ${checkpoint_dir} ${checkpoint_name}
  model=${checkpoint_dir}/${model}
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
    echo "Decoding ..."
    gpuid_list_array=(${gpuid_list//,/ })
    for JOB in $(seq ${nj}); do
        {
        id=$((JOB-1))
        gpuid=${gpuid_list_array[$id]}
        mkdir -p ${output_dir}/output.$JOB
        python infer.py \
            --model ${model} \
            --text_in ${output_dir}/split/text.$JOB.scp \
            --output_dir ${output_dir}/output.$JOB \
            --model_revision ${model_revision}
            --gpuid ${gpuid}
        }&
    done
    wait

    mkdir -p ${output_dir}/final_res
    if [ -f "${output_dir}/output.1/infer.out" ]; then
      for i in $(seq "${nj}"); do
          cat "${output_dir}/output.${i}/infer.out"
      done | sort -k1 >"${output_dir}/final_res/infer.out"
    fi
fi

