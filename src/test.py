from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, get_scheduler

e5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
print(e5_tokenizer.vocab_size)
lm_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
print(lm_model)
# lm_model.resize_token_embeddings(e5_tokenizer.vocab_size)
# print(lm_model.lm_head.out_features)
lm_model.lm_head.out_features = e5_tokenizer.vocab_size
print(lm_model)