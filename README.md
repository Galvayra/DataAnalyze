# 과제 1. Action 데이터 분석 실행 가이드

이 프로젝트는 `datasets/action`의 CSV 데이터를 기반으로 SQLite 데이터베이스를 만들고, 1번부터 5번까지의 분석 결과를 HTML로 생성합니다.

## DB 생성 방법

먼저 CSV 데이터를 SQLite로 적재해야 합니다.

```bash
python3 build_action_db.py
```

생성 위치:

- SQLite 파일: `sqlite/action_analysis.sqlite`

## 분석 실행 방법

공통 실행 형식:

```bash
python3 analyze_actions.py --analysis <번호>
```

### 1. 시간대별 이벤트와 구매 흐름

시간대별 이벤트 수, 주문 수, 매출, 구매전환 흐름을 비교해 구매 가능성이 높은 시간대를 파악하기 위한 분석입니다.

```bash
python3 analyze_actions.py --analysis 1
```

### 2. 카테고리별 주문 추이

카테고리별 관심도, 주문 수, 구매전환율, 매출을 비교해 어떤 카테고리가 실제 성과로 이어지는지 파악하기 위한 분석입니다.

```bash
python3 analyze_actions.py --analysis 2
```

### 3. 연령대별 주문 추이

연령대별 주문 수, 구매자 비율, 시간대별 주문 분포, 선호 카테고리를 비교해 연령별 구매 패턴을 파악하기 위한 분석입니다.

```bash
python3 analyze_actions.py --analysis 3
```

### 4. 검색 키워드와 구매 전환

검색 결과에서 상품을 조회한 행동 이후 실제 구매로 이어진 정도를 키워드별로 비교해 검색 효율을 파악하기 위한 분석입니다.

```bash
python3 analyze_actions.py --analysis 4
```

### 5. 쇼핑몰 타깃 연령과 실제 구매 연령 비교

쇼핑몰이 설정한 타깃 연령과 실제 구매 연령대를 비교해 타깃 적합성과 노출 전략 조정 포인트를 파악하기 위한 분석입니다.

```bash
python3 analyze_actions.py --analysis 5
```

## HTML 출력 위치

모든 분석 결과는 아래 폴더에 생성됩니다.

- `analyze/output/`

생성 파일 예시:

- `analyze/output/time_of_day_orders.html`
- `analyze/output/category_order_trends.html`
- `analyze/output/age_order_trends.html`
- `analyze/output/search_keyword_conversion.html`
- `analyze/output/shop_target_age_comparison.html`

---

# 과제 2. 상품 카테고리 분류 실행 가이드

이 작업은 `datasets/category/items.csv`를 기반으로 상품명과 상세설명을 전처리한 뒤, `title + description` 입력으로 상품 카테고리를 분류하는 분류 모델을 학습하고 평가합니다.

## 전처리 개요

`preprocess.py`는 다음 작업을 수행합니다.

- 원본 상품 CSV를 읽어 `pickle/items.pkl`에 item dictionary 캐시 생성
- 카테고리 목록을 `parsing/categories.json`에 저장
- 상품명은 괄호/품번/특수문자를 정리하여 `title`로 정규화
- 상세설명은 HTML 태그, URL, 반복 특수문자, 불필요한 기호를 제거하여 `description`으로 정규화
- 전체 데이터를 `train`, `valid`, `test` 비율로 분할하여 JSON 파일 생성
- `train` 기준으로 TF-IDF vocabulary 크기를 함께 출력

생성 파일:

- `pickle/items.pkl`
- `parsing/categories.json`
- `parsing/train.json`
- `parsing/valid.json`
- `parsing/test.json`

## 전처리 실행 방법

기본 전처리 실행:

```bash
python3 preprocess.py
```

이미 `parsing/train.json`, `parsing/valid.json`, `parsing/test.json`이 존재하면 해당 파일들을 다시 로드해서 사용합니다.

## represent 생성 옵션

상세설명이 충분히 존재하는 상품에 대해 LLM 기반 재서술 텍스트를 추가하려면 `--add_represent` 옵션을 사용할 수 있습니다.

```bash
python3 preprocess.py --add_represent
```

이 옵션을 사용하면 다음 흐름으로 동작합니다.

- `pickle/represent_items.pkl`을 우선 사용
- 파일이 이미 존재하면 이를 로드
- 파일이 없으면 원본 item 정보를 바탕으로 `represent`를 생성한 뒤 저장
- `prompt/represent.py`의 프롬프트를 사용해 `gpt-oss-120b` vLLM 서버를 호출
- 생성된 결과는 각 item의 `represent` 필드에 저장

## 분류 모델 개요

`classify_categories.py`는 `parsing/train.json`, `parsing/valid.json`, `parsing/test.json`을 읽어 카테고리 분류 실험을 수행합니다.

현재 기본 입력은 `title + description`이며, 주요 실험 구성은 다음과 같습니다.

- Word TF-IDF
  - 단어 단위 unigram, bigram 사용
- Character TF-IDF
  - `char_wb` 기준 3~5그램 사용
- 결합 feature
  - word bigram + char n-gram 결합
- 분류기
  - Linear SVM

## 분류 실행 방법

현재 분류 실험은 아래 5가지 흐름으로 확장되어 있습니다.

### 1. 베이스라인 선정

TF-IDF 기반 feature를 사용해 SVM 분류 성능을 비교합니다.

- Word TF-IDF
  - 단어 단위 unigram, bigram 사용
- Word + Character TF-IDF
  - `char_wb` 기준 3~5그램 사용

실험 결과는 `word TF-IDF`와 `word + char TF-IDF`를 비교해, 어떤 feature 구성이 더 나은지 확인하는 베이스라인으로 사용합니다.

```bash
python3 classify_categories.py --use-tfidf
```

저장 위치:

- `outputs/classification/`

### 2. 임베딩 활용

먼저 SentenceTransformer 임베딩과 FAISS 인덱스를 생성합니다.

```bash
python3 indexing.py --model-name {MODEL_PATH}
```

이 과정에서 다음이 생성됩니다.

- `pickle/embedding/` 하위 embedding pkl
- `index/` 하위 FAISS index 및 item_id 매핑 파일

이후 생성된 embedding pkl을 사용해 두 가지 분류 실험을 수행합니다.

- 임베딩값만 사용한 SVM 분류
- `word + char TF-IDF` sparse vector에 embedding dense vector를 concat한 SVM 분류

```bash
python3 classify_categories.py --use-embeddings
```

저장 위치:

- `outputs/classification_with_emb/`

### 3. FFNN 모델 학습 후 적용

사전 계산된 embedding pkl을 입력으로 FFNN 분류기를 학습합니다.

```bash
python3 run_train.py --model-name {MODEL_PATH} --hidden-dims 256,256,128,128
```

주요 옵션:

- `--hidden-dims 256,256,128,128`
  - hidden layer 크기를 콤마(`,`)로 구분해 지정합니다.
  - 예: `--hidden-dims 512,256,128`

이 과정에서:

- 학습된 checkpoint는 `outputs/ckpt/<ckpt_name>/` 아래에 저장
- 학습 요약은 `outputs/ckpt/<ckpt_name>/training_summary.json`에 저장

이후 저장된 `best_valid_macro_f1.pt` 체크포인트를 불러와 test split 평가를 수행할 수 있습니다.

```bash
python3 classify_categories.py --use-ffnn
```

저장 위치:

- `outputs/classification_with_ffnn/`

### 4. Semantic Search 기반 분류

FAISS 인덱스에서 검색한 top-k 후보의 라벨을 reciprocal-rank voting으로 집계해 카테고리를 예측합니다.

```bash
python3 classify_categories.py --use-search
```

저장 위치:

- `outputs/classification_with_search/`

### 5. Semantic Search + LLM 기반 분류

FAISS 인덱스에서 검색한 top-3 후보 문서를 LLM 프롬프트에 함께 넣어, 전체 후보 카테고리 중 하나를 선택하도록 분류합니다.

```bash
python3 classify_categories.py --use-rag
```

저장 위치:

- `outputs/classification_with_rag/`

## 공통 저장 파일 예시

각 실험 출력 폴더에는 아래와 같은 파일들이 생성됩니다.

- `metrics_summary.csv`
- `experiment_ranking.json`
- `data_overview.json`
- `*_classification_report.json`
- `*_confusion_matrix.csv`
- `*_predictions.csv`

## 참고

- 과제 1은 사용자 행동 데이터 EDA 작업입니다.
- 과제 2는 상품 텍스트 기반 카테고리 분류 작업입니다.
- 두 작업은 서로 독립적인 별도 과제로 관리합니다.
