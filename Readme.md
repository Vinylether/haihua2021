# 初赛

参考技术组baseline： [multiple_choice_baseline_41.9](https://www.biendata.xyz/models/category/6359/) by Zheng_Heng

修改自transformers官方[example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/multiple-choice)

使用预训练模型[hfl/chinese-macbert-large](https://huggingface.co/hfl/chinese-macbert-large)

#### How It Works

在transformers的examples中有一个multiple-choice用例，用于多（单）项选择，仅需少许修改即可使用：

- 将数据转换为swag格式
- 在run_swag.py中加入预测部分 (do_predict)
- 微调参数  略

本项目的run_swag.py文件来自https://github.com/LogicJake/competition_baselines/tree/master/competitions/haihua2021

#### How To Use

***requirements*** :

`pandas`

`tqdm`

`torch`

`transformers`

`datasets`

`sentencepiece`

`protobuf`

将preprocess_data.py, run_swag.py与train.json, validation.json至于同一目录下，新建空文件夹output以存放输出

1. 运行preprocess_data.py将数据转为swag格式，得到train.csv, test.csv

2. 运行run_swag.py以进行训练与预测

   ```bash
   python run_swag.py \
   --model_name_or_path 'hfl/chinese-macbert-large' \
   --do_train \
   --do_predict \
   --max_seq_length 512 \
   --train_file train.csv \
   --test_file test.csv \
   --learning_rate 2e-5 \
   --num_train_epochs 10 \
   --output_dir 'output' \
   --gradient_accumulation_steps 8 \
   --per_device_eval_batch_size 8 \
   --per_device_train_batch_size 2 \
   --overwrite_output
   ```

在output目录下得到模型文件，部分checkpoint，以及输出test_results.txt

初赛时设置文本截断的最大长度为1024，batch_size为2，使用单块RTX 3090训练10个epochs，共用时24492.6194s，得分63.1675874769797

PS：该代码未对复赛提交进行适配

