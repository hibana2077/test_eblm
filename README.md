# test_eblm

## Run training

```bash
python3 main.py --batch_size 4 --num_epochs 5 --dataset_name "wikitext" --dataset_config_name "wikitext-2-raw-v1" --learning_rate 1e-4
```

or

```bash
python3 main.py --batch_size 4 --num_epochs 5 --dataset_name "opencsg/chinese-fineweb-edu" --learning_rate 1e-4
```

## Run inference

```bash
python3 inference.py --input_text "兩個人聊天聊久了都會滋生" --checkpoint_dir "./e5-smollm-model/checkpoint-2" --max_length 150
```