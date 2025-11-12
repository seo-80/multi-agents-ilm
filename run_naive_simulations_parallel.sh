#!/bin/bash
set -euo pipefail

# --- Python interpreter ---
PYTHON="${PYTHON:-$HOME/.pyenv/versions/3.11.6/envs/study-env/bin/python}"

# --- 設定ファイルの読み込み ---
# Load configuration from config.yaml via Python helper
eval "$($PYTHON scripts/load_config.py)"

# --- デフォルト設定 ---
PYTHON_SCRIPT="src/naive_simulation.py"

COUPLING_STRENGTHS=(0.01)
ALPHA_PER_DATA_LIST=(0.001)
N_I_LIST=(100)
FLOW_TYPES=("outward" "bidirectional")
NONZERO_ALPHAS=("center" "evenly")

AGENTS_COUNT=5
MAX_T=1000000
# MAX_T=20000000

usage() {
  cat <<EOF
Usage: $0 [-p param_name] [--param-pair param1,param2] [-m max_t] [-P] [-A] [--dry-run] [--resume]
  -p, --param     変化させるパラメータ (strength | alpha | Ni) — 省略時は Ni, strength, alpha の順で全スイープ
  --param-pair    2つのパラメータを同時にsweep (例: strength,Ni または alpha,Ni)
  -m, --max-t     MAX_T の値（デフォルト: $MAX_T）
  -P, --plot      plotnaive_simulation.py を実行（--plot_distance --skip 1000）
  -A, --animation make_animation.py を実行（アニメーション生成モード）
  --start-idx     アニメーション開始インデックス（デフォルト: 900）
  --end-idx       アニメーション終了インデックス（デフォルト: 1000）
  --interval      アニメーション間隔（ミリ秒、デフォルト: 200）
  --cmap-type     カラーマップタイプ（continuous | discrete、デフォルト: discrete）
  --dry-run       投入コマンドの表示のみ
  --resume        既存CSVの最終t>=MAX_Tはスキップ（未完のみ再実行）
  -h, --help
EOF
  exit 1
}

# --- オプション解析 ---
PLOT_MODE=false
ANIMATION_MODE=false
DRY_RUN=false
RESUME=false
PARAM_TO_CHANGE=""
PARAM_PAIR=""
START_IDX=900
END_IDX=1000
INTERVAL=200
CMAP_TYPE="discrete"

while [[ $# -gt 0 ]]; do
  case "$1" in
  -p | --param)
    PARAM_TO_CHANGE="$2"
    shift 2
    ;;
  --param-pair)
    PARAM_PAIR="$2"
    shift 2
    ;;
  -m | --max-t)
    MAX_T="$2"
    shift 2
    ;;
  -P | --plot)
    PYTHON_SCRIPT="src/plotnaive_simulation.py"
    PLOT_MODE=true
    shift
    ;;
  -A | --animation)
    PYTHON_SCRIPT="src/make_animation.py"
    ANIMATION_MODE=true
    shift
    ;;
  --start-idx)
    START_IDX="$2"
    shift 2
    ;;
  --end-idx)
    END_IDX="$2"
    shift 2
    ;;
  --interval)
    INTERVAL="$2"
    shift 2
    ;;
  --cmap-type)
    CMAP_TYPE="$2"
    shift 2
    ;;
  --dry-run)
    DRY_RUN=true
    shift
    ;;
  --resume)
    RESUME=true
    shift
    ;;
  -h | --help) usage ;;
  --)
    shift
    break
    ;;
  -*)
    echo "Unknown option: $1" >&2
    usage
    ;;
  *) break ;;
  esac
done

# --- sweep 設定（未指定なら3種類すべて実行） ---
PARAMS_TO_RUN=()
PAIR_MODE=false

if [[ -n "$PARAM_PAIR" ]]; then
  # パラメータペアモード
  PAIR_MODE=true
  IFS=',' read -r PARAM1 PARAM2 <<<"$PARAM_PAIR"
  echo "2パラメータ同時sweep: ${PARAM1} と ${PARAM2}"
  PARAMS_TO_RUN=("${PARAM1},${PARAM2}")
elif [[ -n "$PARAM_TO_CHANGE" ]]; then
  # 単一パラメータモード
  PARAMS_TO_RUN=("$PARAM_TO_CHANGE")
else
  echo "パラメータ未指定: Ni, strength, alpha の順で全スイープを実行"
  PARAMS_TO_RUN=("Ni" "strength" "alpha")
fi

echo "===== シミュレーション開始 (Task Spooler) ====="
echo "MAX_T = $MAX_T"
mkdir -p results

# --- extra args (plot mode / animation mode) ---
EXTRA_ARGS=()
if $PLOT_MODE; then
  # 距離に加えて年齢の可視化も実行
  # KDE計算は非常に遅いため、--skip_kdeを追加
  EXTRA_ARGS+=(--plot_distance --plot_similarity --skip 1000 --skip_kde)
elif $ANIMATION_MODE; then
  # アニメーション生成用のパラメータ
  EXTRA_ARGS+=(--start_idx "$START_IDX" --end_idx "$END_IDX" --interval "$INTERVAL" --cmap_type "$CMAP_TYPE" --animation_type "agent")
fi

# パラメータ設定関数
set_parameter_values() {
  local param_name="$1"
  case "$param_name" in
  strength) COUPLING_STRENGTHS=(0.0025 0.005 0.01 0.02 0.04) ;;
  alpha) ALPHA_PER_DATA_LIST=(0.00025 0.0005 0.001 0.002 0.004) ;;
  Ni) N_I_LIST=(1 25 50 100 200 400) ;;
  *)
    echo "エラー: 無効なパラメータ名。strength | alpha | Ni" >&2
    usage
    ;;
  esac
}

run_param_grid() {
  # 現在の配列内容を使用してTSVを出力（tsp実行の元データ）
  for strength in "${COUPLING_STRENGTHS[@]}"; do
    for alpha_data in "${ALPHA_PER_DATA_LIST[@]}"; do
      for flow in "${FLOW_TYPES[@]}"; do
        for nz_alpha in "${NONZERO_ALPHAS[@]}"; do
          for ni in "${N_I_LIST[@]}"; do
            printf "%s\t%s\t%s\t%s\t%s\n" \
              "$strength" "$alpha_data" "$flow" "$nz_alpha" "$ni"
          done
        done
      done
    done
  done
}

for TARGET in "${PARAMS_TO_RUN[@]}"; do
  # 既定値へ戻してから対象のみ拡張
  COUPLING_STRENGTHS=(0.01)
  ALPHA_PER_DATA_LIST=(0.001)
  N_I_LIST=(100)

  if $PAIR_MODE; then
    # パラメータペアモード
    IFS=',' read -r PARAM1 PARAM2 <<<"$TARGET"

    # 両方のパラメータを設定
    set_parameter_values "$PARAM1"
    set_parameter_values "$PARAM2"

    echo "--- スイープ開始: ${PARAM1} と ${PARAM2} の同時sweep ---"
  else
    # 単一パラメータモード
    set_parameter_values "$TARGET"

    echo "--- スイープ開始: ${TARGET} ---"
  fi

  # ログ/結果/スクリプトディレクトリ（tsp用に簡易ログ運用）
  if $PAIR_MODE; then
    # パラメータペアモードの場合、2つのパラメータ名を組み合わせたディレクトリ名
    RESULT_DIR="results/${PARAM1}_${PARAM2}"
  else
    # 単一パラメータモードの場合
    RESULT_DIR="results/${TARGET}"
  fi
  LOG_DIR="${RESULT_DIR}/logs"
  SCRIPT_DIR="${RESULT_DIR}/scripts"
  mkdir -p "$LOG_DIR" "$SCRIPT_DIR"

  # 送信ジョブ数カウンタ
  ENQUEUED=0

  # TSVを読み取り、1件ずつtspに投入
  while IFS=$'\t' read -r strength alpha_data flow nz_alpha ni; do
    # Python側の保存サブディレクトリ命名に合わせて、完了/再開判定用のパスを構築
    if [[ "$flow" == "bidirectional" ]]; then
      flow_prefix="bidirectional_flow-"
    else
      flow_prefix="outward_flow-"
    fi
    subdir="${flow_prefix}nonzero_alpha_${nz_alpha}_fr_${strength}_agents_${AGENTS_COUNT}_N_i_${ni}_alpha_${alpha_data}"
    RAW_DIR="${DATA_RAW_DIR}/${subdir}"

    # --resume: 既存CSVの最終tを確認し、MAX_T以上ならスキップ（アニメーションモードでは適用しない）
    if $RESUME && ! $ANIMATION_MODE; then
      if [[ -f "${RAW_DIR}/save_idx_t_map.csv" ]]; then
        last_t=$(awk -F, 'NR>1{t=$2} END{if (t=="") t=0; print t}' "${RAW_DIR}/save_idx_t_map.csv" 2>/dev/null || echo 0)
        if [[ "$last_t" =~ ^[0-9]+$ && "$last_t" -ge "$MAX_T" ]]; then
          echo "[SKIP] 完了済み: ${subdir} (t=${last_t} >= ${MAX_T})"
          continue
        fi
      fi
    fi

    # ログファイル
    safe_subdir=${subdir//\//_}
    LOG_FILE="${LOG_DIR}/${safe_subdir}.log"

    # コマンドラインを安全にクォートしてラッパースクリプトを生成
    args=(
      "$PYTHON" "$PYTHON_SCRIPT"
      "--coupling_strength" "$strength"
      "--alpha_per_data" "$alpha_data"
      "--flow_type" "$flow"
      "--nonzero_alpha" "$nz_alpha"
      "--N_i" "$ni"
      "--agents_count" "$AGENTS_COUNT"
      "--recompute_distance"
    )
    # アニメーションモード以外では --max_t を追加
    if ! $ANIMATION_MODE; then
      args+=("--max_t" "$MAX_T")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      args+=("${EXTRA_ARGS[@]}")
    fi
    printf -v cmd_quoted '%q ' "${args[@]}"

    SCRIPT_FILE="${SCRIPT_DIR}/${safe_subdir}.sh"
    cat >"$SCRIPT_FILE" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
${cmd_quoted}>"$LOG_FILE" 2>&1
SCRIPT
    chmod +x "$SCRIPT_FILE"

    if $DRY_RUN; then
      echo "tsp \"$SCRIPT_FILE\""
    else
      tsp "$SCRIPT_FILE"
    fi
    ENQUEUED=$((ENQUEUED + 1))
  done < <(run_param_grid)

  if $PAIR_MODE; then
    echo "--- スイープ完了: ${PARAM1} と ${PARAM2} の同時sweep（投入ジョブ数: ${ENQUEUED}） ---"
  else
    echo "--- スイープ完了: ${TARGET}（投入ジョブ数: ${ENQUEUED}） ---"
  fi
done

echo "===== 全てのスイープが完了しました ====="
