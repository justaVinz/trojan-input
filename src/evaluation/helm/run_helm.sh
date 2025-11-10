helm-run \
  --conf-paths ./run_specs/run_entries_core_scenarios_10.conf \
  --suite v1 \
  --max-eval-instances 10 \
  --enable-local-huggingface-models ./../../../models/gpt2 \
  --models-to-run openai/gpt2 \
  --local-path ./config

helm-summarize \
  --suite mmlu \
  --output-path ./benchmark_output/

helm-server --suite mmlu