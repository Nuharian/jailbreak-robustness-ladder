# convenience runner (also runnable on Windows via: bash run_all.sh)
set -e
M="python -m jbladder.run_eval --keep-raw --max-new-tokens 160"
$M --model Qwen/Qwen2.5-0.5B-Instruct        --batch-size 32
$M --model Qwen/Qwen2.5-1.5B-Instruct        --batch-size 32
$M --model HuggingFaceTB/SmolLM2-1.7B-Instruct --batch-size 32
$M --model Qwen/Qwen2.5-3B-Instruct          --batch-size 12
$M --model Qwen/Qwen2.5-1.5B-Instruct --defense system --batch-size 32
$M --model HuggingFaceTB/SmolLM2-1.7B-Instruct --defense system --batch-size 32
python -m jbladder.analyze
