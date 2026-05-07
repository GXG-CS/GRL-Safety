#!/usr/bin/env bash
# Submit isolated OFA-full prompt-graph runs from this worktree.

set -euo pipefail

PROJECT=${OFA_PROJECT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}
RUNNER=sgb/models/gfm/ofa/run_ofa.py
LOG_DIR="$PROJECT/sgb/models/gfm/ofa/logs"
mkdir -p "$LOG_DIR"
RUN_TAG=${OFA_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OFA_OUT_DIR:-$PROJECT/experiments/ofa/runs/$RUN_TAG}
mkdir -p "$OUT_DIR"

MODE=${1:?mode required: smoke|probe|probe_full|clean|fn|ed|imb|struct|fair|interp|ood_degree|ood_time|all}; shift || true

submit_one() {
    local axis=$1
    local ds=$2
    local extra="${3:-} ${OFA_EXTRA:-}"

    local mem=48G
    local time=03:00:00
    local constraint=""
    local train_bs=256
    local eval_bs=256
    local max_train=0
    local out_csv="$OUT_DIR/${axis}.csv"
    local per_class_csv=""

    case "$ds" in
        arxiv|arxiv23|arxivyear|amazonratings|bookhis|bookchild|sportsfit|elecomp|tolokers)
            mem=96G
            time=06:00:00
            constraint="--constraint=a100"
            train_bs=128
            eval_bs=128
            ;;
        pubmed|wikics|dblp|elephoto)
            mem=64G
            time=04:00:00
            constraint="--constraint=a100"
            train_bs=192
            eval_bs=192
            ;;
    esac

    case "$axis" in
        interp)
            max_train=512
            train_bs=32
            eval_bs=32
            extra="$extra --n_seeds 5 --max_test_nodes 100"
            ;;
        imb|struct|ood_degree|ood_time)
            max_train=1024
            ;;
    esac

    if [[ "$axis" == "imb" ]]; then
        per_class_csv="$OUT_DIR/imb_per_class.csv"
    fi

    local job=ofa_${axis}_${ds}
    local slurm=$LOG_DIR/${job}.slurm
    cat > "$slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --output=$LOG_DIR/${job}_%j.log
#SBATCH --error=$LOG_DIR/${job}_%j.log
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${mem}
#SBATCH --time=${time}
${constraint:+#SBATCH $constraint}

set -euo pipefail

cd $PROJECT
eval "\$(conda shell.bash hook)"
conda activate safety
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GFM_PROJECT_ROOT=$PROJECT
date
echo "=== OFA-full ${axis} ${ds} ==="

python -u $RUNNER \
  --axis ${axis} \
  --dataset ${ds} \
  --epochs 100 \
  --patience 30 \
  --lr 1e-3 \
  --dropout 0.15 \
  --num_layers 7 \
  --num_rels 5 \
  --emb_dim 768 \
  --train_query_batch_size ${train_bs} \
  --eval_query_batch_size ${eval_bs} \
  --max_train_query ${max_train} \
	  --gpu 0 \
	  --output_csv ${out_csv} \
	  ${per_class_csv:+--per_class_csv ${per_class_csv}} \
	  ${extra}

date
EOF
    local sbatch_args=()
    if [[ -n "${OFA_DEPENDENCY:-}" ]]; then
        sbatch_args+=(--dependency="$OFA_DEPENDENCY")
    fi
    sbatch "${sbatch_args[@]}" "$slurm"
}

if [[ "$MODE" == "smoke" ]]; then
    submit_one clean cora "--n_splits 1 --epochs 30 --patience 10 --max_train_query 512"
    submit_one clean pubmed "--n_splits 1 --epochs 30 --patience 10 --max_train_query 512"
elif [[ "$MODE" == "probe" ]]; then
    submit_one clean cora "--n_splits 1 --epochs 100 --patience 30 --max_train_query 0"
    submit_one clean pubmed "--n_splits 1 --epochs 100 --patience 30 --max_train_query 0"
elif [[ "$MODE" == "probe_full" ]]; then
    submit_one clean cora "--n_splits 1 --epochs 100 --patience 30 --max_train_query 0 --prompt_graph_mode full"
    submit_one clean pubmed "--n_splits 1 --epochs 100 --patience 30 --max_train_query 0 --prompt_graph_mode full"
elif [[ "$MODE" == "clean" ]]; then
    if [[ $# -gt 0 ]]; then
        datasets=("$@")
    else
        datasets=(cora pubmed)
    fi
    for ds in "${datasets[@]}"; do submit_one clean "$ds"; done
elif [[ "$MODE" == "all" ]]; then
    for ds in cora citeseer pubmed wikics dblp arxiv23 elecomp elephoto amazonratings sportsfit; do
        submit_one clean "$ds"
    done
    for ds in cora pubmed; do
        submit_one fn "$ds"
        submit_one ed "$ds"
        submit_one imb "$ds"
        submit_one struct "$ds"
        submit_one interp "$ds"
    done
    submit_one fair tolokers
else
    datasets=("$@")
    if [[ ${#datasets[@]} -eq 0 ]]; then
        datasets=(cora pubmed)
    fi
    for ds in "${datasets[@]}"; do submit_one "$MODE" "$ds"; done
fi
