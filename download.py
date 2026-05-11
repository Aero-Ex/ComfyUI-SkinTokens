from huggingface_hub import hf_hub_download, snapshot_download

import argparse

REPO_ID = "VAST-AI/SkinTokens"

MODELS = [
    "experiments/skin_vae_2_10_32768/last.ckpt",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
]

DATASETS = [
    "rignet.zip",
    "articulation.zip",
]

LLM_REPO = "Qwen/Qwen3-0.6B"
LLM_LOCAL_DIR = "models/Qwen3-0.6B"


def download_model(name: str):
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=name,
        local_dir=".",
    )
    print(f"[MODEL] {name} downloaded to: {local_path}")


def download_llm():
    local_path = snapshot_download(
        repo_id=LLM_REPO,
        local_dir=LLM_LOCAL_DIR,
        ignore_patterns=["*.bin", "*.safetensors"],
    )
    print(f"[LLM] Config downloaded to: {local_path}")


def download_data(name: str):
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"dataset_clean/{name}",
        local_dir=".",
    )
    name = name.removesuffix(".zip")
    local_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"datalist/{name}/*"],
        local_dir=".",
    )
    print(f"[DATA] {name} downloaded to: {local_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true", help="Download model checkpoints")
    parser.add_argument("--data", action="store_true", help="Download datasets")
    args = parser.parse_args()
    if not args.model and not args.data:
        print("Please specify --model or --data")
        return
    if args.model:
        for model in MODELS:
            download_model(model)
        download_llm()
    if args.data:
        for data in DATASETS:
            download_data(data)


if __name__ == "__main__":
    main()
