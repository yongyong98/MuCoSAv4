# PAIR-BST 독립 seed별 complete-OOF classification protocol

Protocol ID: `cv3_independent_seed_oof_v1`

## 적용 범위

이 protocol은 PAIR-BST 7-model benchmark의 canonical classification
평가법이다. 모델 7개, ROI 표현 3개, classification task 3개의 모든 조합에
적용한다. Retrieval protocol은 변경하지 않는다.

고정된 patient-disjoint fold와 결과 label은 변경하지 않는다. Operational
patient identifier는 임상팀이 확인한 diagnosis와 `patient_idx`의 composite를
그대로 사용한다. 향후 data team이 제공할 globally unique patient identifier와의
대조는 별도의 외부 확인 사항으로 남긴다.

## 추정량

고정 seed는 `101`, `202`, `303`, `404`, `505`이다. 각 seed마다 독립적으로
학습된 head를 세 held-out fold에 평가한다. 세 held-out subset을 frozen ROI
위치에 다시 배치하여 seed당 하나의 완전한 2,252-ROI OOF prediction set을 만든다.

Balanced accuracy, macro-F1, accuracy, weighted-F1은 seed별 complete OOF에서
각각 한 번 계산한다. Primary 결과는 이렇게 얻은 다섯 metric의 산술평균과
sample standard deviation이다. Sample standard deviation은 `ddof=1`을 사용한다.

Canonical path에서는 seed 간 probability나 logit을 평균하지 않는다. Hard
vote를 사용하지 않으며 15개의 seed-and-fold metric을 독립 primary observation으로
취급하지 않는다. Fold metric은 audit evidence로만 보존한다.

## 재구성 원자료

Canonical v4 결과는 완료된 live run의 암호학적으로 검증된 189개
`seed_and_mean_probabilities.npz`에서 재구성한다. Canonical prediction에는
`seed_probabilities`, `test_indices`, 검증된 seed order만 사용한다. 저장된
`mean_probabilities`는 legacy audit data이며 canonical output의 입력이 될 수 없다.

각 시스템에서 class order, fold assignment, metadata order, probability dimension,
finite value, held-out index의 상호 배타성, 전체 coverage, seed별 ROI당 prediction
1개를 검증한다. Linear head를 다시 학습하지 않고 frozen feature를 다시 추출하지
않으며 retrieval도 다시 실행하지 않는다.

## Canonical 산출물

- `classification_seed_oof_metrics.csv`: 315행
- `classification_seed_fold_metrics.csv`: audit용 945행
- `classification_seed_oof_predictions.csv.gz`: 709,380행
- `classification_seed_oof_probabilities/`: system별 compressed file 63개
- `classification_per_class_seed_oof.csv`: 5,250행
- `classification_per_class_seed_summary.csv`: 1,050행
- seed별 및 seed summary confusion-matrix file
- `classification_patient_cluster_ci_by_seed.csv`: seed별 supplementary interval
- 315행 seed OOF metric file만 입력으로 사용하는 Table 5

Patient-cluster confidence interval은 seed마다 별도로 계산하며 하나의 인위적인
interval로 합치지 않는다. Seed 5개만으로 새로운 canonical paired significance
claim을 만들지 않는다. 기존 ensemble 기반 paired analysis는 명확히 표시된
legacy evidence로만 보존한다.

## Version 분리

기존 v2와 v3 package는 변경하지 않는 legacy 및 audit record이다. 이들의
classification estimator는 fold 내 probability를 평균하고 fold 간 변동을
보고했다. 해당 package의 classification file은 이 protocol의 canonical input으로
사용할 수 없다. Protocol ID, estimator field, source hash, output hash, row-count
검사와 retrieval hash가 모두 일치해야 v4 validation을 통과한다.

