# List of models
models=(
    "deffuant"
    "hk_averaging"
    "carpentras"
    "duggins"
)

for model in "${models[@]}"
do
  echo "Producing baseline figure for $model"
  python plotting/plot_figure.py --model "$model"

  echo "Producing optimized figure for $model"
  if [ "$model" == "duggins" ]; then
    python plotting/plot_figure.py --model "$model" --experiment "optimized" --filetype "jsonl"
  else
    python plotting/plot_figure.py --model "$model" --experiment "optimized"
  fi

  echo "Producing noise figure for $model"
  if [ "$model" == "duggins" ]; then
    python plotting/plot_figure.py --model "$model" --experiment "noise" --filetype "jsonl"
  else
    python plotting/plot_figure.py --model "$model" --experiment "noise"
  fi
done
