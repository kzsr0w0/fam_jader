
# モジュールのインポート
import pandas as pd

# --- データの準備 --------------------------------------------- #
demo_df = pd.read_csv('C:\\folder\\make\\Python\\jader\\data\\demo202311-01.csv', encoding='cp932')
drug_df = pd.read_csv('C:\\folder\\make\\Python\\jader\\data\\drug202311-01.csv', encoding='cp932')
reac_df = pd.read_csv('C:\\folder\\make\\Python\\jader\\data\\reac202311-01.csv', encoding='cp932')

# データフレームのマージ
df = pd.merge(demo_df, drug_df, on = '識別番号')
df = pd.merge(df, reac_df, on='識別番号')

# 明らかに関係のない列の削除
drop_col = ['識別番号', '報告回数_x','報告年度・四半期', '状況', '報告の種類',
       '報告者の資格', 'E2B', '報告回数_y', '医薬品連番', '医薬品の関与','報告回数', '有害事象連番']
df = df.drop(drop_col, axis=1)

# 対象薬の選択　今回はファモチジンだけにしぼる
fam_df = df[df['医薬品（一般名）'] == 'ファモチジン']
fam_df.head(3)

# --- 採用する説明変数を絞る ------------------------------------- #

# 削除する列を記述
drop_col = ['医薬品（一般名）', '医薬品（販売名）', '経路', '投与開始日', '投与終了日',
       '投与量', '投与単位', '分割投与回数', '使用理由', '医薬品の処置', '再投与による再発の有無', 'リスク区分等',
       '有害事象','有害事象の発現日']
fam_df_dr = fam_df.drop(columns=drop_col, inplace=False)

# 英語にしておく
fam_df_dr = fam_df_dr.rename(columns={'性別': 'gender', '年齢': 'age', '体重': 'bw', '身長': 'height'})


# --- 欠損値の処理 最頻値で埋める ---------------------------------------------- #

for column in ['gender', 'age', 'bw', 'height']:
    mode_value = fam_df_dr[column].mode()[0]
    fam_df_dr[column].fillna(mode_value, inplace=True)

# --- 数値への変換 ------------------------------------------------ #

# 転帰が不明の症例を除外
fam_df_dr = fam_df_dr[fam_df_dr['転帰'] != '不明']

# 年齢が数値以外の症例を除外
fam_df_dr = fam_df_dr[~fam_df_dr['age'].isin(['不明','小児','高齢者'])]  
# 年齢が10歳未満を9歳代に変換
fam_df_dr['age'] = fam_df_dr['age'].replace('10歳未満','9歳代')
# 年齢を数値に変換
fam_df_dr['age'] = fam_df_dr['age'].str.replace('歳代','').astype(int)

# 体重が10kg未満を9kg台に変換
fam_df_dr['bw'] = fam_df_dr['bw'].replace('10kg未満','9kg台')
# 体重を数値に変換
fam_df_dr['bw'] = fam_df_dr['bw'].str.replace('kg台','').astype(int)

# 身長を数値に変換
fam_df_dr['height'] = fam_df_dr['height'].str.replace('cm台','').astype(int)

# --- ワンホットエンコーディング --------------------------------------- #

en = pd.get_dummies(fam_df_dr['gender'])
fam_df_dr['gender'] = en['女性']

# --- 機械学習 --------------------------------------------------- #

# 目的変数
t = fam_df_dr['転帰']
# 入力変数
x = fam_df_dr.drop('転帰', axis=1)

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=86, max_depth=11, random_state=0)

# モデルの学習
model.fit(x, t)


# -----------------#

# モデルの保存
import pickle
pickle.dump(model, open('model_fam','wb'))
