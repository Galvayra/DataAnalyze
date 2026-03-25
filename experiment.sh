#!/usr/bin/env bash

set -euo pipefail

# models 폴더 하위의 모든 모델 디렉터리를 자동으로 수집
MODELS_DIR="models"
MODEL_PATHS=()

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Model directory not found: $MODELS_DIR" >&2
  exit 1
fi

while IFS= read -r model_dir; do
  MODEL_PATHS+=("$model_dir")
done < <(for path in "$MODELS_DIR"/*; do
  if [[ -d "$path" ]]; then
    printf '%s\n' "$path"
  fi
done | sort)

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "No model directories found under: $MODELS_DIR" >&2
  exit 1
fi


# 1. 모델별 임베딩 생성 및 FAISS 인덱스 구축
# 현재 스크립트 이름은 index.py가 아니라 indexing.py 입니다.
for model_path in "${MODEL_PATHS[@]}"; do
  python3 indexing.py --model-name "$model_path"
done

# 2. 모델별 FFNN 분류기 학습
for model_path in "${MODEL_PATHS[@]}"; do
  python3 run_train.py --model-name "$model_path"
done

# 3. 모델별 평가, 만약 vllm으로 llm을 띄워서 연결하지 않았다면 rag 옵션 빼고 실행
python3 classify_categories.py --use-tfidf --use-embeddings --use-ffnn --use-search
#python3 classify_categories.py --use-tfidf --use-embeddings --use-ffnn --use-search --use-rag
