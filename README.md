# W-SDA

**RoPE → Summer Delta Attention → NoPE** の自己回帰言語モデルを、
異なるGPUを持つ自宅DockerとColabなどのワーカーで協調学習する実験用Pythonライブラリです。
MIT / Python 3.10+ / PyTorch。事前学習済みモデルではありません。

> **重要:** CPU上のモデル・勾配・HTTP統合テストを実施した研究用実装です。
> 実Colab GPU、自宅GPU、実WAN、Dockerイメージの起動はこのリリースで実機検証していません。
> SDAは未融合の逐次PyTorch実装です。大規模LLMを高速に学習できるという実証はありません。

## アーキテクチャ

```text
Colab / 自己管理GPU                 自宅Docker GPU
  完全なモデル複製                    完全なモデル複製
  ローカルのデータ shard 1             ローカルのデータ shard 0
  複数stepのAdamW                     複数stepのAdamW
       │ INT8パラメータ差分                │
       └────────── HTTPS ─────────────────┘
                     │
           CPU coordinator（自宅/VPS）
           staleness-aware model mixing
           SQLite + safetensors + 冪等receipt
```

**系列/ヘッドをWAN越しに分割しません。KVやDelta StateもWANへ送りません。**
各サンプルのforward/backwardは一台で完結し、パラメータ更新だけ低頻度で交換します。
各ワーカーには同一モデル全体とoptimizer/activationが収まる必要があります。
これはVRAMの合算、FSDP、DDPと同値な勾配同期ではありません。

モデルは`layer_pattern=["rope","sda","nope"]`を`num_layers`まで繰り返します。
RoPEはRoPE層のQ/Kだけに適用。SDA/NoPE層には位置埋め込みを追加しません。
SDAは再帰順序を持ち、NoPEでも因果マスクは維持します。SwiGLU、RMSNorm、weight tyingを使用。
`ModelConfig`で層順序を変更できます。

### SDA / SDA+

`configs/tiny.json`はSDA-5のdelta更新とhard-gated residual correctionを使用します。
`configs/tiny-plus.json`ではSDA+のtop-m softmaxとsoft confidence gate、INT8 fake quantizationを使用します。

上流: [SummerDeltaAttention](https://github.com/Summer110622/SummerDeltaAttention)、
[SDA+ reference](https://github.com/Summer110622/SummerDeltaAttention/blob/main/summer_delta_attention/plus.py)。
**完全移植ではありません。** 上流のreservoir archive/cluster indexを、学習可能な因果FIFO/archive full scanに変更しています。
学習時のarchive fake quantizationはSTEであり、アクティベーションをINT8で保存する機能ではありません。
一方、WAN差分のINT8圧縮は実際にint8バイト列を送信します。[差分と数式](docs/ARCHITECTURE.md)。

## Colabについて

Googleの[公式FAQ](https://research.google.com/colaboratory/faq.html)では、
**無料のmanaged runtime / 正のcompute unit残高がない環境でのdistributed computing workersが禁止**されています。
有料プランで正の残高がある環境、許可されたEnterprise/専用VM、または自己管理local runtimeで使用してください。
有料でも切断・利用上限がなくなるわけではありません。keep-aliveや制限回避は実装していません。

[Colabのlocal runtime](https://research.google.com/colaboratory/local-runtimes.html)はColabの画面から自分の計算機を操作する機能です。
**ColabのGPUが追加されるわけではありません。** Colab GPU + 自宅GPUの二台にするには、許可されたホスト型GPUが別途必要です。

## 1. インストールとデータ作成

```bash
# private repoの場合は認証済みのgit/ghでclone、または配布ZIPを展開
python -m pip install -e '.[dev]'

# corpus.txtは自分が使用権を持つUTF-8テキスト
wsda prepare --input corpus.txt --out data
wsda configure --template configs/tiny.json --data data --out run.json
```

デフォルトのUTF-8 byte tokenizerは動作確認用です。本格的な学習にはローカルのtokenizer.jsonを指定します。
ネットワーク側へデータやtokenizer本体は自動送信されません。

```bash
python -m pip install -e '.[tokenizer]'
wsda prepare --input corpus.txt --tokenizer tokenizer.json --out data
wsda configure --template configs/tiny-plus.json --data data --out run.json
```

`prepare`の出力先は空ディレクトリにしてください。末尾10%をvalidationへ分離します。
tokenizerのSHA256と語彙数をcoordinatorと照合します。全ノードで同一tokenizerを使用してください。
この例では**同じ準備済みdataディレクトリ**を両ノードに配置し、互いに異なる`shard_index`を使います。
別々のコーパスを使う場合は各ノード`--num-shards 1 --shard-index 0`とし、非IIDによる品質変化を評価してください。

## 2. 自宅Docker

NVIDIAドライバ・NVIDIA Container Toolkit・Docker Composeが必要です。
[GPU設定の公式ドキュメント](https://docs.docker.com/compose/how-tos/gpu-support/)。
デフォルトイメージはPyTorch 2.10.0 / CUDA 12.8 wheelを使用するため、対応するドライバが必要です。

```bash
# 秘密値をソース、ノートブック、gitに保存しない
python -c 'import secrets,pathlib; p=pathlib.Path(".env"); assert not p.exists(),".env already exists"; p.write_text("WSDA_TOKEN="+secrets.token_urlsafe(32)+"\nHOME_BATCH_SIZE=1\n"); p.chmod(0o600)'

docker compose up -d --build coordinator
docker compose --profile gpu up home-worker
```

coordinatorはCPUで動作。自宅ワーカーはGPUを一台使用します。初期設定は100ジョブで終了します。
`data/`は読み取り専用、checkpointとoutboxはnamed volume `wsda-state`へ保存。
`.env`のtokenはColab側でも同じものを入力します。秘密値を画面共有しないでください。

### WAN用HTTPSを有効化

`.env`に`WSDA_DOMAIN=train.your-domain.example`を追加し、DNSとルーターの転送を設定します。
Caddyが証明書を発行する例は外部TCP 80/443が到達可能であることが前提です。
[HTTPSの前提条件](https://caddyserver.com/docs/automatic-https)。

```bash
docker compose -f compose.yaml -f compose.tls.yaml up -d coordinator caddy
```

APIの8000番をインターネットへ直接公開しないでください。デフォルトではlocalhostのみ公開します。
CGNAT等で自宅への着信ができない場合、公開HTTPSを持つ自分のVPS等へcoordinatorを配置し、両ワーカーからそこへ送信します。
本ライブラリはNAT越え、VPS契約、DNS、証明書の自動手配は行いません。
TLS検証は有効、HTTP redirect追従は無効です。`--allow-http`はDocker内部等の信頼できる閉域だけで使用します。

## 3. Colabワーカー

[`notebooks/colab_worker.ipynb`](notebooks/colab_worker.ipynb)をColabで開きます。
リポジトリがprivateでも使えるよう、ソースZIPと準備済みデータZIPをアップロードする方式です。

```bash
# 自宅で実行。data.zipには学習データが含まれるので扱いに注意
python -m zipfile -c data.zip data
```

ノートブックでは、許可された実行環境であることを確認し、APIのHTTPS URLとtokenを入力して実行します。
データは`/content/data`、Colab側は`shard_index=1,num_shards=2`。
GPUごとにmicrobatch/gradient accumulationを変えられます。学習時間のEWMAから次のローカルstep数を調整します。

通常のPython環境では同じ処理をCLIで実行できます。

```bash
# WSDA_TOKENを環境変数に設定（値をコマンド履歴へ直書きしない）
wsda worker --server https://train.your-domain.example \
  --data data --worker-id colab --device cuda \
  --batch-size 1 --grad-accum 2 --jobs 10 \
  --num-shards 2 --shard-index 1
```

`auto`精度は対応GPUでBF16、それ以外でFP32です。T4等では`--precision fp16`を明示指定できます。
CUDA OOMではmicrobatchを減らしaccumulationを増やして、同じeffective batchでジョブをやり直します。
モデル自体が収まらない場合は停止します。モデル構造や系列長をノードごとに変えることはできません。

## 4. 確認・復元・推論

```bash
# ホストのWSDA_TOKENを設定したターミナルで
wsda status --server http://127.0.0.1:8000

# Docker内でcoordinatorの保存済み状態をexportし、ホストへ取り出す
docker compose exec coordinator wsda export --state /state/coordinator --out /state/export
docker compose cp coordinator:/state/export ./export
wsda evaluate --checkpoint export --data data
wsda generate --checkpoint export --prompt 'summer' --max-new-tokens 32
# 独自tokenizerの場合 generateにも --tokenizer data/tokenizer.json を追加
```

coordinatorを同じstate volumeで再起動すれば、全精度モデル・version・receipt・ノード統計を復元します。
送信前の差分はワーカーのoutboxへfsyncし、送信失敗後は同じ`work_dir`で起動すると再送します。
同じ更新を二重適用しません。`work_dir`のないColab VM消失では未送信計算は失われますが、受理済み更新はサーバーに残ります。
ローカル学習の途中やAdamWモーメントの厳密再開は未実装です。rebase時にAdamWをリセットします。

## 学習アルゴリズムと調整

各ワーカーはモデルversion `b`から`K_i`回のAdamWを行い、`delta_i=theta_local-theta_b`を送ります。
coordinatorの現versionを`v`、古さを`s=v-b`として、`s > max_staleness`なら拒否します。
それ以外は`alpha=server_alpha/(1+s)`で以下を計算します。

```text
theta_next = (1-alpha) * theta_current + alpha * (theta_base + clipped_delta)
```

これはFedAsync-styleの非同期モデル混合です。**DiLoCoを再現したものではありません。**
速いノード・頻繁に参加するノードのデータが過大に反映され得ます。
トークン数は記録しますが、厳密なサンプル数重み付きFedAvgではありません。
IIDに近いshardから始め、`evaluate`の共通validation lossで品質を監視してください。

`local_steps`が初回のstep数、以降はtoken/sのEWMAにより`min_local_steps..max_local_steps`内で調整します。
`target_seconds`は目標ローカル計算時間。通信が遅いと`4 × 通信時間`も考慮します。
これは通信率20%を目指すヒューリスティックで、上限により達成できない場合があります。
`server_alpha`を下げると混合が弱くなり、`max_staleness`を下げると遅いノードの更新が捨てられやすくなります。

Pパラメータなら、1ジョブのpayloadは概ね**download 4P bytes + upload P bytes + scale/header**です。
INT8はアップロードのみ。ダウンロードはexact FP32で、更新差分の基点を一致させます。
従って「通信全体が4分の1」ではありません。FP32双方向の8Pに対して約5Pです。
実際のbyte数を`metrics.jsonl`に記録します。

## テスト

```bash
pytest
python scripts/smoke_http.py
```

モデル因果性、独立NumPy式との一致、finite-difference gradient check、通信量子化、重複/期限切れ/古い更新、
tokenizer照合、認証、データshard、実backwardをテストします。
HTTPテストは**本物のlocalhost HTTPとCPUワーカー2プロセス**で学習し、サーバー再起動も検証します。
結果は[`artifacts/http_smoke_results.json`](artifacts/http_smoke_results.json)。
短い合成コーパスのloss低下は動作確認であり、LLM品質・WAN高速化の証拠ではありません。

## 制限

SDAのCUDA/Triton融合kernel、長系列並列scan、WAN tensor/pipeline parallelism、remote KV retrieval、
pretrained Hugging Faceモデルへの自動変換、LoRA、分散optimizer shard、検証済み大規模収束は未実装です。
NoPE/RoPE層はdense attentionなので、このモデル全体は線形attentionではありません。
coordinatorは単一障害点で、各モデルのFP32コピー・直近version群を保持します。大規模モデル向けのストリーミングは未実装。
信頼済みノード専用で、悪意ある参加者のmodel poisoning、差分からの情報漏洩、GPU資源保証には対処していません。
[セキュリティと運用](docs/OPERATIONS.md)を参照してください。
