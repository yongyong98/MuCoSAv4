# PAIR-BST

## 골·연부조직 종양을 위한 영역 단위 조직병리 데이터셋

**저자:** Gyu Yeong Kim, Yongjun Jeon, Hoyeon Jeong, Donggeon Lee, Seungkyun Lee, Hyungbin Kim, Yurimi Lee, Jihwan Kim, Seog-yun Park, Kyu-Hwan Jung, David Joon Ho, June Hyuk Kim, Yoon-La Choi

**소속 기관:** Samsung Medical Center · Sungkyunkwan University · National Cancer Center · The State University of New York, Korea

PAIR-BST 데이터셋 benchmark의 공식 코드 저장소입니다.

[데이터셋](https://doi.org/10.25452/figshare.plus.c.8223469) | 논문: 출판 후 링크 제공 | [English](README.md)

![PAIR-BST의 33개 조직학적 진단 범주를 보여주는 대표 H&E 조직병리 이미지](docs/assets/pair-bst-figure-1.jpg)

*Figure 1. PAIR-BST의 33개 조직학적 진단 범주를 대표하는 H&E 염색 영역.*

## 논문 공개 상태

PAIR-BST는 *PAIR-BST: A region-level histopathology dataset for bone and soft
tissue tumors* 원고에 기술되어 있습니다. 논문은 아직 출판되지 않았으며,
출판 후 공식 논문 링크, 서지정보와 BibTeX를 이 저장소에 추가할 예정입니다.

## 연구 개요

골·연부조직 종양은 희귀하고 형태학적으로 매우 다양하여 조직병리 진단이
어렵습니다. PAIR-BST는 기존 공개 benchmark에 충분히 포함되지 않았던 이들
종양을 대상으로 computational pathology 모델을 개발하고 평가할 수 있도록
구축되었습니다.

데이터셋은 268명 환자의 470개 whole-slide image(WSI)에서 얻은 2,252개의
병리의사 주석 ROI로 구성됩니다. 각 ROI는 4096 x 4096 pixel PNG이며 다음
세 단계의 label을 가집니다.

- 조직학적 진단 33개
- 분화 계통 11개
- 성장 양상 6개

이 저장소는 환자 간 누출이 없는 교차검증 환경에서 frozen feature 기반
linear probing을 수행합니다. 7개 feature extractor와 3개 ROI 표현 전략을
세 분류 과제에 적용하며, 환자 분리 image retrieval, provenance 검증,
불확실성 계산과 논문용 표 생성 기능도 제공합니다.

## 데이터셋

PAIR-BST 데이터는 Figshare+에서 제공합니다.

**https://doi.org/10.25452/figshare.plus.c.8223469**

Figshare collection에는 WSI, 추출된 ROI 이미지와 metadata가 포함됩니다.
ROI benchmark에는 2,252개 PNG와 대응 metadata를 사용합니다. 최신 파일
목록, license와 재사용 조건은 Figshare record를 확인하십시오.

GitHub 저장소는 code-only release입니다. 원본·파생 이미지, checkpoint,
H5 feature, prediction, 로컬 경로 설정과 가명 식별자가 포함된 fold manifest는
공개하지 않습니다.

## Benchmark protocol

| 항목 | Canonical 설정 |
| --- | --- |
| Feature extractors | ResNet-50, Swin-T, RetCCL, UNI, UNI-2, Prov-GigaPath, Virchow2 |
| 과제 | 진단 33-class, 분화 계통 11-class, 성장 양상 6-class |
| ROI 표현 | 224 x 224 center crop, 16 x 16 grid mean pooling, 16 x 16 grid max pooling |
| 평가 | 진단 층화, 환자 분리 3-fold cross-validation |
| Fold 환자 수 | 90, 89, 89명; 모든 fold에 세 과제의 전체 class 포함 |
| Linear probe | Frozen feature, train-only 표준화, weighted cross-entropy, AdamW, 10 epochs |
| Seeds | 101, 202, 303, 404, 505 |
| Primary metrics | seed별 전체 OOF 예측의 balanced accuracy와 macro-F1 |
| 보고 | seed metric 5개의 평균과 표본표준편차 |

동일 환자의 모든 ROI와 WSI는 반드시 같은 fold에 배정됩니다. Fold별 metric은
감사용으로 보존하며, 주요 결과는 각 seed의 전체 out-of-fold 예측에서
계산합니다.

전체 protocol은
[`configs/protocol_cv3_independent_seed_oof_v1.yaml`](configs/protocol_cv3_independent_seed_oof_v1.yaml)에
고정되어 있습니다.

## 저장소 구성

```text
PAIR-BST/
|-- configs/                 # 경로, 모델, 비교 및 protocol 설정
|-- docs/                    # Protocol 및 공개 범위 문서
|-- locks/                   # 비식별 dataset/model contract
|-- scripts/                 # 검증 및 재구성 utility
|-- src/pairbst/             # Benchmark 구현
|-- tests/                   # Unit/integration test
|-- pyproject.toml
`-- requirements.mucosa-cu128.lock.txt
```

## 설치

기준 실행환경은 Python 3.11.13, CUDA 12.8용 PyTorch 2.8.0, NVIDIA RTX
3090입니다. Python 3.11 또는 3.12가 필요하며, 환경이 다르면 해당 platform에
맞는 PyTorch build를 사용하십시오.

```bash
git clone https://github.com/yongyong98/PAIR-BST.git
cd PAIR-BST
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.mucosa-cu128.lock.txt
python -m pip install -e .
pairbst --help
```

Linux 또는 macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.mucosa-cu128.lock.txt
python -m pip install -e .
pairbst --help
```

## 데이터 및 checkpoint 설정

실행 전에 machine-local 설정 파일을 만듭니다.

```powershell
# Windows PowerShell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
Copy-Item locks/EXECUTION_HOLD.example.json locks/EXECUTION_HOLD.json
```

```bash
# Linux 또는 macOS
cp configs/paths.example.yaml configs/paths.local.yaml
cp locks/EXECUTION_HOLD.example.json locks/EXECUTION_HOLD.json
```

`configs/paths.local.yaml`에서 다음 경로를 실제 위치에 맞게 수정하십시오.

- Figshare metadata CSV와 4096 x 4096 ROI PNG directory
- 7개 feature extractor의 local checkpoint
- local output 및 lock directory

Checkpoint는 이 저장소에서 배포하지 않습니다. 각 모델의 공식 upstream
source와 license/access 조건을 따라 획득해야 합니다. 모델 identity, feature
dimension, transform, revision과 SHA-256은
[`configs/models.yaml`](configs/models.yaml) 및
[`locks/models.expected.json`](locks/models.expected.json)에 기록되어 있습니다.

## 실행 방법

다음 명령은 canonical 기본 설정을 사용합니다.

### 1. 입력 준비 및 검증

```powershell
pairbst manifest build --verify-dimensions
pairbst splits build
pairbst images verify-release --workers 4
pairbst models verify
```

필수 데이터, model identity 또는 checksum이 맞지 않으면 실행이 중단됩니다.
`images verify-release`에는 `paths.local.yaml`에 지정된 release file manifest가
필요합니다.

### 2. 실행 계획 확인

```powershell
pairbst features extract --model all --dry-run
```

### 3. Deterministic pilot

Local audit가 통과하고 dataset/model 사용 조건을 확인한 경우에만
`--override-hold`를 사용하십시오.

```powershell
pairbst features pilot --model all --override-hold
```

### 4. 7-model 전체 benchmark

다음은 모든 prerequisite가 준비된 재현 workspace를 위한 명령입니다.
실제 실행에는 다운로드한 데이터와 7개 checkpoint 외에도 생성된 manifest와
split lock, image-integrity record, deterministic model-pilot record 및 고정된
`locks/environment.current.json`이 필요합니다. Recovery 전용 audit와
`scripts/reconstruct_independent_seed_oof.py`에는 공개하지 않은 내부 보존
artifact가 추가로 필요합니다.

```powershell
$Tag = "pairbst_7model_v1"

pairbst features extract `
  --model all `
  --output-dir "outputs/runs/features/$Tag" `
  --override-hold

pairbst classify run `
  --model all `
  --features-dir "outputs/runs/features/$Tag" `
  --output-dir "outputs/runs/classification/$Tag" `
  --override-hold

pairbst retrieval run `
  --model all `
  --features-dir "outputs/runs/features/$Tag" `
  --output-dir "outputs/runs/retrieval/$Tag" `
  --override-hold

pairbst statistics run `
  --classification-dir "outputs/runs/classification/$Tag" `
  --retrieval-dir "outputs/runs/retrieval/$Tag" `
  --output-dir "outputs/runs/statistics/$Tag" `
  --override-hold

pairbst report build `
  --classification-dir "outputs/runs/classification/$Tag" `
  --retrieval-dir "outputs/runs/retrieval/$Tag" `
  --statistics-dir "outputs/runs/statistics/$Tag" `
  --output-dir "outputs/final/$Tag" `
  --override-hold
```

생성된 feature, 분류·retrieval 결과, 통계, provenance와 CSV/Markdown/LaTeX
표는 Git에서 제외되며 승인된 local storage에 보관해야 합니다.

## 테스트

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -q
```

다운로드한 Figshare 데이터나 통제된 split-audit fixture가 필요한 integration
test는 해당 artifact가 없으면 skip됩니다.

## 재현성과 공개 범위

- Dataset/model contract는 SHA-256으로 고정됩니다.
- Recursive provenance 검사로 서로 다른 run의 artifact 혼합을 차단합니다.
- Train/held-out fold 사이의 환자 및 WSI 중복을 허용하지 않습니다.
- Primary benchmark에는 모델별 공식 preprocessing을 사용합니다.
- 결과, checkpoint와 가명 row-level 파일은 public GitHub에 포함하지 않습니다.

자세한 내용은 [`docs/PUBLIC_RELEASE_BOUNDARY.md`](docs/PUBLIC_RELEASE_BOUNDARY.md)와
[`docs/INDEPENDENT_SEED_OOF_PROTOCOL_KO.md`](docs/INDEPENDENT_SEED_OOF_PROTOCOL_KO.md)를
참고하십시오.

## 논문 및 인용

논문은 아직 출판되지 않았습니다. 출판 후 공식 논문 URL과 BibTeX를 제공할
예정입니다. 그전에는
[Figshare+ record](https://doi.org/10.25452/figshare.plus.c.8223469)에서
제공하는 dataset citation을 사용해 주십시오.

## 이용 조건

데이터에는 Figshare+ record에 표시된 license와 이용 조건이 적용됩니다.
Pretrained checkpoint에는 각 upstream source의 license 및 access 조건이
적용됩니다. 코드 license는 추후 release에서 추가할 예정입니다.

## 문의

코드 또는 재현 관련 문의는 GitHub issue를 이용해 주십시오.
