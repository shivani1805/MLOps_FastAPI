from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data, get_metrics


app = FastAPI()

class WineData(BaseModel):
    alcohol: float
    flavanoids: float
    color_intensity: float
    proline : int

class WineResponse(BaseModel):
    target_class:int

class MetricResponse(BaseModel):
    accuracy : float
    precision : float
    recall : float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.get("/model_info", status_code=status.HTTP_200_OK)
async def model_info():
    return {
        "model": "wine_decision_tree",
        "features": ["alcohol", "flavanoids", "color_intensity", "proline"],
        "description": {
            "alcohol": "Alcohol content of the wine",
            "flavanoids": "Flavonoid content, related to phenolic compounds",
            "color_intensity": "Visual color intensity of the wine",
            "proline": "Proline amino acid content"
        }
    }


@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        features = [[wine_features.alcohol, wine_features.flavanoids,
                    wine_features.color_intensity, wine_features.proline]]

        prediction = predict_data(features)
        return WineResponse(target_class=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/metrics", response_model=MetricResponse)
async def evaluate_metric():
    acc, precision, recall = get_metrics()
    return MetricResponse(accuracy=acc, precision=precision, recall=recall)

    


    