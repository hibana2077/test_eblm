# test_eblm

Run

```bash
python3 main.py --batch_size 4 --num_epochs 5 --dataset_name "wikitext" --dataset_config_name "wikitext-2-raw-v1"
```

```bash
python3 inference.py --input_text "Hi I am" --checkpoint_dir "./e5-smollm-model" --max_length 150
```