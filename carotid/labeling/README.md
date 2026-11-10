# 라벨링 안내 (batch01, 78장)

## 1. Label Studio 실행

```bash
cd ~/Documents/2026_LifeStep_홍석찬/carotid
bash setup/setup_labelstudio.sh
```

브라우저에서 `http://localhost:8080`이 열린다. 처음에는 계정을 만든다(내 Mac 안에만 저장).

## 2. 프로젝트 만들기

1. **Create Project** → 이름 `carotid_batch01`
2. **Data Import** → 처음에는 `labeling/batch01/first10/` 의 PNG **10장만** 끌어다 놓기 (시간 측정용, 4절)
3. **Labeling Setup** → **Custom template** → `labeling/label_config.xml` 내용을 붙여넣기 → **Save**
4. 10장을 끝내고 시간을 잰 뒤, 같은 프로젝트에 `labeling/batch01/rest68/` 의 68장을 추가로 가져온다.
   (`images/` 폴더에는 78장 전체가 그대로 있다. 이미 가져온 10장을 다시 넣지 않도록 주의)

## 3. 그리는 방법

- **경동맥 내강만** 다각형으로 그린다. 정맥은 그리지 않는다(배경).
- 위쪽 벽의 **안쪽(아래쪽) 경계**를 왼쪽 → 오른쪽으로 찍고, 아래쪽 벽의 **안쪽(위쪽) 경계**를 오른쪽 → 왼쪽으로 찍어 닫는다.
- 혈관이 화면 좌우 밖으로 나가면 **화면 끝까지만** 그린다. 화면 밖은 추측하지 않는다.
- 밝기·대비 조절과 확대를 써도 된다.
- 세트를 반드시 고른다(2026-11-10부터 A/B/C/F/X). 여러 개에 해당하면 더 나쁜 쪽(F > C > B > A).

| 세트 | 기준 | 다각형 |
|---|---|---|
| A | 위·아래 벽의 **내벽 흰 띠**까지 선명, 혈관이 화면 좌우 끝까지 이어짐 | 보이는 내강을 화면 끝까지 |
| B | 내강은 보이지만 내벽 흰 띠가 안 보이거나 흐림 | 보이는 내강 |
| C | 위·아래 벽은 보이지만 혈관이 **화면 중간에서 끝나거나 끊김** | 보이는 내강만 |
| F | 위·아래 벽 중 **한쪽이 안 보이거나** 경동맥이 **아예 안 보임**(횡단면·탐촉자 이동) | 확실한 내강만. 없으면 **그리지 않음**(빈 마스크로 학습에 사용) |
| X | 동맥·정맥 구분을 확신할 수 없음 | 그리지 않음, 제외 사유 선택 |

- 동맥과 정맥이 함께 보여도 **동맥이 확실하면 그린다**. 어느 쪽이 동맥인지 모르면 X.
- 화면 끝까지 이어진 혈관은 흰 띠 선명도로 A/B를 나눈다. C는 혈관이 화면 **안에서** 끝나거나 끊긴 경우다(2026-11-10 확정).
- 이미 batch01·02를 예전 A/B/C 기준으로 그렸다면, Label Studio의 **Labeling Setup**에 새 `label_config.xml`을 붙여넣고 세트만 다시 고른다.

## 4. 처음 10장으로 시간 재기

처음 10장(P001 3장, P002 3장, P003 4장)을 먼저 끝내고 내보내기 → 변환하면 장당 작업 시간이 나온다.

- 시간은 Label Studio가 장마다 자동 기록한다(작업 화면을 연 시점부터 제출까지). 한 장을 열어 둔 채 쉬면 시간이 늘어나므로, 쉴 때는 제출하거나 건너뛴 뒤 쉰다.
- 처음 1~2장은 도구에 익숙해지는 시간이 섞이므로 결과에는 평균과 함께 중앙값도 나온다.
- 결과(`reports/labeling_batch01.txt`)에 전체 목표 515장(학습·검증·시험 495 + 반복 20)의 예상 시간이 같이 나온다. 이 숫자를 라벨 일정 근거로 쓴다.

## 5. 내보내기와 변환

1. 프로젝트 화면 → **Export** → **JSON** → 다운로드
2. 터미널에서:

```bash
cd ~/Documents/2026_LifeStep_홍석찬/carotid
source .venv/bin/activate
python tools/ls_export_to_masks.py ~/Downloads/<내보낸파일>.json
```

- 마스크: `data/labels/<영상ID>/<프레임>_mask.png`
- 세트·사유·메모: `manifest/frames.csv`
- 결과 요약과 장당 작업 시간: `reports/labeling_batch01.txt`
- 여러 번 내보내서 다시 실행해도 된다(최신 라벨로 덮어씀).
- 내보낸 JSON은 `labeling/<배치>/exports/<배치>_<내용>_<날짜>.json`으로 옮겨 보관한다(예: `batch01/exports/batch01_first10_20261110.json`). 라벨 원본 기록이므로 지우지 않는다.

## 6. batch02 (새 8명, 150장)

batch01이 끝난 뒤 같은 방법으로 진행한다.

1. **Create Project** → 이름 `carotid_batch02`, 설정은 batch01과 같은 `label_config.xml`
2. **Data Import** → `labeling/batch02/images/` 의 PNG 150장
3. 내보낸 뒤 변환할 때 배치 이름을 붙인다:

```bash
python tools/ls_export_to_masks.py ~/Downloads/<내보낸파일>.json --batch batch02
```

| 구분(2026-11-10 기준) | 사람 | 장수 | 주의 |
|---|---|---:|---|
| 학습 | P005·P007·P008·P010·P011 | 각 15 | P005의 V17 3장은 종단면이 오른쪽 위에 일부만 보임 → 보통 B |
| 학습 | P009 | 30 | 원래 시험용으로 뽑았으나 학습으로 옮김. 그리는 방법은 같다 |
| 검증 | P006 | 15 | 경동맥이 화면 **위쪽**(넓은 간격의 두 줄 사이). 아래의 밝은 굴곡 구조물과 헷갈리지 않기 |
| 시험 | P004 | 30 | 평가 전용. 모델 초안 없이 처음부터 직접 그린다 |

- P005(V18·V19)도 P006과 비슷하게 경동맥이 위쪽에 있는 배치로 보인다. 위치보다 **평행한 두 벽**을 기준으로 찾는다.
- P007은 회전 촬영이라 경계가 흐린 장면이 많다. 벽 한쪽이 안 보이면 F, 동맥인지 모르겠으면 X.

## 7. batch03 (추가 267장) · 반복 라벨 (20장)

batch01·02 라벨이 끝난 뒤 진행한다.

```bash
python tools/select_label_frames.py batch03     # 학습 232 · 검증 25 · 시험 10장 (F 후보 36장 포함)
python tools/select_label_frames.py repeat01    # A·C로 라벨된 프레임 중 20장 (사람별로 고루)
```

- batch03에는 그동안 제외했던 구간(횡단면, 탐촉자 이동, P007 V21 전체)의 프레임이 섞여 있다. 혈관이 안 보이면 **F를 고르고 그리지 않는다.**
- batch03의 시험 P004 10장은 모델 초안 없이 직접 그린다.
- 반복 라벨은 batch01·02를 그린 날과 **다른 날**, 이전 라벨을 보지 않고 그린다. 변환할 때 `--repeat`를 붙이면 `data/labels_repeat/`에 따로 저장되고 원래 라벨은 바뀌지 않는다.

```bash
python tools/ls_export_to_masks.py ~/Downloads/<내보낸파일>.json --batch batch03
python tools/ls_export_to_masks.py ~/Downloads/<내보낸파일>.json --batch repeat01 --repeat
```
