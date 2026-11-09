# VisA PCB1：データ独立性・類似画像スクリーニング調査報告

**対象プロジェクト：** Defect Inspector 2.0（VisA PCB1）\
**関連Issue：** [#1 データ独立性・リーク検証](https://github.com/Rinmq/defect-inspector-visa-portfolio/issues/1)\
**調査時点：** 2026-11-08〜09\
**判定：** 関連画像の独立性は**未確認**。データリークの存在・不存在のいずれも断定しない。

## 1. 目的と対象

評価対象モデルのTrain・Validation・Test間に、見た目の近い画像や同一基板・同一撮影系列に由来する画像が混在する可能性を評価する。目的は、モデルの精度を再調整することではなく、**評価結果の解釈に関する制約を特定すること**である。

対象はVisA `pcb1` の既存の固定分割である。公式Trainから層化抽出したValidationを使用し、公式Testは変更しない。

| 分割 | 画像数 |
|---|---:|
| Train | 529 |
| Validation | 133 |
| Test | 442 |
| **合計** | **1,104** |

**本調査ではモデルの再学習、閾値調整、データ再分割、公式Testを使ったモデル選択を行っていない。**

## 2. 手法と再現手順

既存の [`tools/check_visa_split_similarity.py`](https://github.com/Rinmq/defect-inspector-visa-portfolio/blob/main/tools/check_visa_split_similarity.py) を使用した。このスクリプトは画像をグレースケール化し、32×32に縮小して2次元離散コサイン変換（DCT）を計算する。左上8×8成分のうちDC成分を除いた63成分を中央値と比較し、63ビットのpHashを生成する。

2画像間の距離はハッシュの異なるビット数（ハミング距離）とし、Train/Validation、Train/Test、Validation/Testの全ペアを比較した。分割ごとに距離4以下・8以下のペア数を集計し、距離の小さい各6ペアを抽出した。

使用したローカル実行コマンド（**読み取り対象は変更しない**）：

```powershell
& "C:\defect_inspector_2\.venv\Scripts\python.exe" "C:\defect_inspector_portfolio\tools\check_visa_split_similarity.py" --manifest "C:\defect_inspector_portfolio\data\splits\visa_pcb1_highshot_v1.json" --data-root "C:\defect_inspector_2\data\raw\VisA" --output-dir "C:\defect_inspector_2\data\raw\VisA\_review"
```

調査で生成されたローカル成果物：

- `data/raw/VisA/_review/split_similarity_report.json`：ペア数・pHash距離の集計と候補一覧
- `data/raw/VisA/_review/split_similarity_review.jpg`：代表ペアの目視比較シート
- `data/raw/VisA/_review/033_mask_overlay.png`：代表欠陥画像とマスクの重ね合わせ

これらの成果物は**GitHubに未登録**。スクリプト自体は公開済み。パスはローカルの調査環境におけるものである。

## 3. 全ペアのpHash検査結果

| 比較対象 | ペア総数 | 距離 ≤ 4 | 距離 ≤ 8 | 最小距離 |
|---|---:|---:|---:|---:|
| Train / Validation | 70,357 | 10,580 | 26,358 | 0 |
| Train / Test | 233,818 | 32,956 | 86,107 | 0 |
| Validation / Test | 58,786 | 9,092 | 22,598 | 0 |

すべての分割間で、**pHash距離0の異なる画像パスのペア**が抽出された。距離0は「今回の63ビットpHashが一致した」という意味であり、画素が同じ、同じ物理的な基板である、あるいはデータリークが確定したという意味ではない。

このデータセットでは同型のHC-SR04基板が多数含まれ、全体の構造がよく似る。大きな領域の配置を主に表現するpHashは、その類似性に反応しやすい。一方、微小な欠陥は低周波成分に反映されないことがある。

## 4. 代表画像の詳細調査

### 4.1 Train `Anomaly/033.JPG` とValidation `Normal/0737.JPG`

- 正解ラベル：`defect` と `good`
- pHash距離：**0**
- 目視：全体の基板配置はよく似ているが、欠陥画像では下部のTrig/Echo端子付近に局所的な外観差がある。
- `033.png`の公式マスク：非ゼロ画素 **2,393**、外接矩形 `(xmin=634, ymin=705, xmax=706, ymax=752)`。
- 元画像とマスクの重ね合わせ：注釈領域が下部端子付近に位置することを確認。

このペアは、**pHashが同じでも公式に注釈された局所欠陥が異なる**という具体例である。ただし、同一物理基板かどうかは分からない。

### 4.2 Train `Anomaly/073.JPG` とValidation `Normal/0830.JPG`

- 正解ラベル：`defect` と `good`
- pHash距離：**0**
- 目視：全体構造は類似。欠陥画像の上部金属部品周辺などに外観差が見られた。
- `073.png`の公式マスク：非ゼロ画素 **3,294**、全体の外接矩形 `(xmin=569, ymin=318, xmax=790, ymax=833)`。
- 8連結の連結成分解析：**8領域**。最大領域は **2,806画素**、矩形 `(x=681, y=318, width=110, height=52)`。

連結成分の個数は、独立した8種類の欠陥があることを意味しない。マスクの小領域やノイズの可能性を含む。公式注釈箇所の詳細な画像重ね合わせは、このペアでは未完了。

### 4.3 その他の代表例

- Train `Normal/0004.JPG` とValidation `Normal/0254.JPG` もpHash距離0として抽出された。
- このペアは元解像度での詳細照合を実施していない。

## 5. 撮影系列を識別する情報の確認

- `033.JPG`, `0737.JPG`, `073.JPG`, `0830.JPG` の4画像についてPillowの`getexif()`を確認した結果、**いずれも空**だった。
- `pcb1/image_anno.csv` のヘッダーは `image,label,mask` の3列であり、**基板個体IDや撮影系列IDの列は存在しない**。
- これは上記4画像と当該CSVについての確認であり、データセット全体に撮影系列を識別する手段が一切ないことまでは証明しない。
- 物理的な同一性を裏付ける撮影ログ・個体ID・連続撮影情報は、**本調査では取得できていない**。

## 6. 結論、制約と今後の仮説

### 確認できた事実

1. 既存の分割監査では画像パスの重複は0件と報告されている。本調査はその結果を再検証していない。
2. 3種類の分割間で、pHash距離0の画像ペアが存在する。
3. 正常・欠陥のラベルが異なる画像間でもpHash距離0となる具体例を確認した。
4. `033.JPG`では、pHashが一致する他ラベル画像との比較において、公式マスクに対応した局所欠陥を確認した。
5. 代表4画像のEXIFは空であり、確認したCSVには個体・撮影系列IDの列がない。

### 未確認・断定できないこと

- 同一物理基板が複数の分割に含まれているか。
- 撮影系列単位でTrain/Validation/Testが独立しているか。
- 関連画像によるモデル精度の上振れが、実際にどの程度生じているか。
- pHash距離の小さいペアが、どれだけ同一個体を含むか。

### 暫定的な技術判断

**本調査は、関連画像の存在を疑う根拠とpHashによるスクリーニングの限界を示したが、データリークの有無は立証していない。** そのため、既存の評価値を無効とは断定しない一方、独立した個体・撮影系列への汎化性能と同一視することも避ける。

今後、個体IDや撮影系列IDなどの信頼できる根拠が得られれば、Group-aware分割での再評価を**別の実験計画**として検討できる。ただし、既存の学習済みモデル、分割、公式Testの記録を上書きしない。新しい実験を行う場合はTrain/Validationで設計と選択を行い、公式Testをモデル選択に流用しない。

## 7. 検証の追跡性とIssue完了条件

| Acceptance Criteria | 状態 | 根拠・残件 |
|---|---|---|
| 既存の類似画像検査方法を理解・確認 | 実施 | コードのpHash計算・候補抽出を確認 |
| 分割間の類似画像候補を再現可能に抽出 | 実施 | 公開スクリプトと実行コマンド、集計ログ |
| 代表画像を目視確認 | 実施 | 2組の異ラベルペアを比較 |
| 同一基板・撮影系列の識別可能性を調査 | 一部実施 | EXIF 4枚とCSV列を確認。物理的同一性は未確認 |
| リーク可能性と未確認事項を整理 | 実施 | 本文5・6節 |
| 検証スクリプトと分析レポートを残す | **レポート作成済み、GitHub未反映** | 既存スクリプト公開済み。本書を追加予定 |
| 結果に基づく改善仮説を記録 | 実施 | Group-aware分割の検討。ただし実施許可なしに再学習しない |

**現時点ではIssue #1をCloseしない。** GitHubに本書を追加して内容と根拠をレビューした後、未解決の物理的同一性を制約として受け入れてCloseできるか判断する。

## 8. 参考資料

- [Defect Inspector 2.0公開リポジトリ](https://github.com/Rinmq/defect-inspector-visa-portfolio)
- [Issue #1（調査コメントと議論）](https://github.com/Rinmq/defect-inspector-visa-portfolio/issues/1)
- [`tools/check_visa_split_similarity.py`](https://github.com/Rinmq/defect-inspector-visa-portfolio/blob/main/tools/check_visa_split_similarity.py)
- [`data/splits/visa_pcb1_highshot_v1.json`](https://github.com/Rinmq/defect-inspector-visa-portfolio/blob/main/data/splits/visa_pcb1_highshot_v1.json)
- [READMEの分割・制約説明](https://github.com/Rinmq/defect-inspector-visa-portfolio/blob/main/README.md)

**調査記録の根拠：** 実行ログ、提示された元画像・マスク重ね合わせ画像、Issueコメント。本書の数値はそれらの記録から転記し、新規の全件再実行は行っていない。
