import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

model = pickle.load(open('C:/Users/21261/Desktop/Master 2/Devops and MLops/Lab folder - FastAPI Pydantic/Lab folder - FastAPI Pydantic/model.pkl','rb'))

app = FastAPI()

class furniture(BaseModel):
    category : int
    sellable_online : int
    other_colors : int
    depth : float
    height : float
    width : float

@app.get("/")
def home():
    return {'ML for furniture prediction'}

@app.post("/predictions")
async def predictions(features : furniture):
    pred = model.predict([[features.category,features.sellable_online,features.other_colors,features.depth,features.height,features.width]])
    return str(pred[0])

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)