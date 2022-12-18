# marl-robots-2d
マルチエージェント深層強化学習の実世界ロボットへの適用に向けた
2Dシミュレーション環境

Experiments with deep multi-agent reinfocement learning for robots in a simple 2D environment.

適用アルゴリズム: QMIX

## 使い方

```
python src/main.py --algo qmix --env diamond --wandb
```

`--algo`: 使用するアルゴリズムを指定

`--env`: 使用する環境を指定

`--wandb`: Weights & Biasesに結果を記録


## 実験環境 "Diamond" 

`src/envs/diamond/diamond.py`

エージェント数: 2

巡回する警備員に見つからずにダイアモンドを盗めればクリア

Robot Agent (RA): 周りの障害物をレーザーで観測しながら動くことができる

Sensor Agent (SA): 完全観測能力を持ち、Robot Agentにメッセージを送ることができる

### RAの観測 (Toyota HSRを想定)
- LiDARで観測した距離 * レーザーの本数
- エージェントから見たゴールの相対的な座標 (x, y)
- SAからのメッセージ (One-Hot)

### SAの観測
- RAの絶対位置 (x, y)
- 警備員の絶対位置 (x, y)
- ゴールの相対的な座標 (x, y)



## アルゴリズム

[QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) (
オックスフォード大学, 2018)


論文の著者が実装したレポジトリ [oxwhirl / **pymarl**](https://github.com/oxwhirl/pymarl)


## Docker
```
cd docker
bash build.sh
```
```
bash run.sh
```