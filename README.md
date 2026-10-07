# Monocular Distance Estimation from Object Size

A small neural network that predicts an object's real-world distance from its apparent size in pixels, built around the classical pinhole-camera relationship and trained to recover that relationship from noisy, synthetic sensor-like data.

## Motivation

Range-estimation systems (e.g. single-camera/IR-based distance sensing) often can't rely on a clean geometric formula in practice — real detections are noisy, with imprecise bounding boxes and sensor imperfections. This project demonstrates that a small neural network can learn the underlying inverse relationship between object size and distance implicitly from data, without being given the formula directly, and degrades gracefully in a realistic, explainable way.

## Approach

1. Physics-based synthetic data generation — using the pinhole camera model:
   ```
   pixel_size = (focal_length × real_world_size) / distance
   ```
   500 examples were generated with distances sampled uniformly between 2–100 meters.

2. Realistic noise injection — Gaussian noise (N(0, 5)) was added to pixel sizes to simulate real-world sensor/detection imprecision, rather than training on perfectly clean geometric data.

3. Preprocessing — inputs were standardized (StandardScaler, mean 0 / std 1) before training. This was a critical fix: without scaling, the model collapsed to predicting roughly the dataset average for every input (MAE ≈ 17.3m). After scaling, MAE dropped to ≈ 7.2m with predictions that correctly tracked the true values.

4. Model — a small Multi-Layer Perceptron (MLPRegressor, 2 hidden layers × 16 neurons), trained with scikit-learn.

5. Evaluation — 80/20 train/test split; performance measured with Mean Absolute Error (MAE) and a predicted-vs-true scatter plot.

## Results

- Final MAE: ≈ 7.17 meters across a 2–100m test range.
- Predictions closely track true distances at short range (<30m), where the model is highly accurate.
- Key finding: accuracy degrades at longer range (>50-60m), with the model systematically underpredicting distance. This matches real-world range-estimation behavior: because pixel size changes very little per meter at long distances (the inverse relationship flattens out), a fixed amount of sensor noise represents a much larger relative signal distortion far away than up close — so the model has less reliable signal to work with for distant objects.

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/4fd7c8c2-e5d8-41b6-8c19-45c94be1465c" />

*(scatter plot: predicted vs. true distance, red dashed line = perfect prediction)*

## Tech stack

- Python, NumPy (synthetic data generation)
- scikit-learn (MLPRegressor, StandardScaler, train_test_split, mean_absolute_error)
- Matplotlib (evaluation plot)
- joblib (model persistence)

## How to run

1. Open the notebook in Google Colab (or any Jupyter environment with scikit-learn installed).
2. Run all cells top to bottom — the dataset is generated synthetically, so no external data download is required.
3. The trained model and scaler are saved as distance_model.pkl and scaler.pkl via joblib, and can be reloaded with:
   ```python
   import joblib
   model = joblib.load('distance_model.pkl')
   scaler = joblib.load('scaler.pkl')
   ```

## Possible extensions

- Replace synthetic data with a real dataset (e.g. KITTI or FLIR ADAS Thermal) for distance/depth ground truth.
- Add more input features (e.g. object class, aspect ratio) beyond pixel size alone.
- Try an ensemble or a model explicitly weighted to handle the long-range accuracy drop-off (e.g. loss weighting by distance).
