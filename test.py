from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="haosulab/ManiSkill", repo_type="dataset", allow_patterns="*.json"
)
