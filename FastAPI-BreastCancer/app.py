import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


model = pickle.load(open('modelRForst.pkl', 'rb'))
app = FastAPI()
class Breastcancer(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float



@app.get("/")
def home():
    return {'message': 'ML for Breast Cancer prediction'}

@app.post("/predictions")
async def predictions(features: Breastcancer):
    input_data = [[
        features.radius_mean, features.texture_mean, features.perimeter_mean, features.area_mean,
        features.smoothness_mean, features.compactness_mean, features.concavity_mean, features.concave_points_mean,
        features.symmetry_mean, features.fractal_dimension_mean, features.radius_se, features.texture_se,
        features.perimeter_se, features.area_se, features.smoothness_se, features.compactness_se, features.concavity_se,
        features.concave_points_se, features.symmetry_se, features.fractal_dimension_se, features.radius_worst,
        features.texture_worst, features.perimeter_worst, features.area_worst, features.smoothness_worst,
        features.compactness_worst, features.concavity_worst, features.concave_points_worst, features.symmetry_worst,
        features.fractal_dimension_worst
    ]]
    pred = model.predict(input_data)

    if pred[0] == 0:
        prediction_text = "Maligne"  # 1 est malin
    else:
        prediction_text = " Bénin"  # 0 est bénin

    return {"prediction": prediction_text}


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
