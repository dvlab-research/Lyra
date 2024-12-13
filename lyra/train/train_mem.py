from lyra.train.train import train
from lyra.model.qwen2vl_top_attn import replace_qwen2vl_attn_with_top_attn
replace_qwen2vl_attn_with_top_attn()

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")