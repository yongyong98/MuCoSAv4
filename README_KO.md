# PAIR-BST 벤치마크

이 저장소는 PAIR-BST 개정 실험을 재현하기 위한 정리된 코드입니다.
기존의 오래된 루트 스크립트는 현재 `main` 트리에서 제거했으며, 과거
이력은 Git 기록에 남아 있습니다. 공개 데이터 컬렉션은
[Figshare](https://plus.figshare.com/collections/PAIR-BST_A_region-level_histopathology_dataset_for_rare_bone_and_soft_tissue_tumors/8223469)에서 제공합니다.

## 공개 범위

공개 저장소에는 소스 코드, 이식 가능한 설정 템플릿, 테스트, 문서와
비식별 checksum contract만 포함합니다. 원본·파생 이미지, 모델 가중치,
H5 feature, 실험 출력, credential, 로컬 경로, 가명 환자·ROI 식별자가 든
fold manifest는 포함하지 않습니다. `configs/paths.example.yaml`을 복사해
Git에서 제외되는 `configs/paths.local.yaml`을 만들고, 승인된 로컬 자료만
연결해야 합니다.

## 분류 평가 원칙

- 모델 7개, ROI 전략 3개, 과제 3개로 총 63개 시스템을 평가합니다.
- 고정 seed는 101, 202, 303, 404, 505입니다.
- 각 seed에서 환자 단위 3-fold held-out 예측을 연결해 하나의 완전한 OOF
  예측 집합을 만듭니다.
- balanced accuracy와 macro-F1은 seed별 전체 OOF 예측에서 계산합니다.
- Table 5는 다섯 seed metric의 산술평균과 표본표준편차(`ddof=1`)를
  사용합니다.
- seed 사이의 probability, logit 또는 prediction을 합쳐 투표하지 않습니다.

현재 canonical protocol은 `cv3_independent_seed_oof_v1`입니다. 이전
`PAIRBST-REV-CV3-v1` seed-probability ensemble은 감사용 legacy protocol이며
canonical Table 5 입력으로 사용하지 않습니다.

## 설치

```powershell
python -m pip install -r requirements.mucosa-cu128.lock.txt
python -m pip install -e .
$env:PYTHONPATH = "src"
python -m pairbst.cli --help
```

실제 실행 전에 로컬 데이터, 승인된 fold, model identity와 checksum을
연결하고 감사 절차를 통과해야 합니다. 공개 저장소의
`locks/EXECUTION_HOLD.example.json`은 승인 그 자체가 아니라 안전한 기본
상태의 예시입니다.

## 검증된 Path-A 재구성

보존된 held-out probability NPZ 189개가 있는 내부 환경에서는 linear head
재학습, frozen feature 재추출 또는 retrieval 재실행 없이 canonical 결과를
재구성할 수 있습니다.

```powershell
python scripts/reconstruct_independent_seed_oof.py `
  --classification-root <RETAINED_CLASSIFICATION_RESULT_DIRECTORY> `
  --folds <RETAINED_APPROVED_FOLD_MANIFEST> `
  --config configs/protocol_cv3_independent_seed_oof_v1.yaml `
  --prior-package <RETAINED_VALIDATED_V2_DIRECTORY> `
  --output <NEW_V4_OUTPUT_DIRECTORY>
```

공개 저장소는 원본 fold-level NPZ와 내부 결과 package를 포함하지 않습니다.
재구성에는 승인된 보존 artifact 경로가 별도로 필요합니다.

## 공개 체크아웃 테스트

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -q
```

통제된 split 감사 fixture 또는 Figshare 로컬 데이터가 필요한 integration
test는 해당 자료가 없으면 명시적으로 skip됩니다. 공개 범위와 보안 원칙은
`docs/PUBLIC_RELEASE_BOUNDARY.md`를 참고하십시오.
