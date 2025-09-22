## Dataset Used for This Lab
**UCI Wine Dataset:** [https://archive.ics.uci.edu/dataset/109/wine](https://archive.ics.uci.edu/dataset/109/wine)
- The dataset consists of **13 features**, **178 instances** and **3 target classes**.  
- For simplicity, only **4 features** are being used to train the Gaussian Naive Bayes model in this lab depending on the importance of the features:  
  1. `alcohol`  
  2. `flavanoids`  
  3. `color_intensity`
  4. `proline`

## Target Classes
The model predicts one of the following 3 classes of wine:
- 0
- 1
- 2

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
    ```bash
   cd src
   python train.py 
   ```

3. Run the API server:
    ```bash
    uvicorn main:app --reload
    ```
4. Test endpoints:
- /predict - predict class for a single sample
- /metrics - view accuracy, precision & recall
- For OpenAPI specification, use [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Example Prediction:

**Request JSON:**
```json
{
  "alcohol": 12.37,
  "flavanoids": 1.05,
  "color_intensity": 1.25,
  "proline": 520
}
```
**Response JSON:**
```json
{
  "target_class": 1
}
```