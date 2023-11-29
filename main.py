
# モジュールのインポート
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle

# インスタンス化
app = FastAPI()

# 入力するデータの型の定義
class Jadar(BaseModel):
    gender: int
    age: int
    bw: int
    height: int

# 学習済みモデルの読み込み
model = pickle.load(open('model_fam','rb'))

# トップページ
@app.get('/')
async def index():
    return{'model_fam':'jader_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def make_predictions(features: Jadar):
    prediction = model.predict([[features.gender, features.age, features.bw, features.height]])[0]
    return {'prediction': str(prediction)}