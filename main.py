
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

if __name__ == "__main__":
    dataset  = fetch_california_housing()
    features, labels = dataset["data"], dataset["target"]
            
    reg = RandomForestRegressor()
    reg.fit(features, labels)
    score = reg.score(features, labels)
    print(f"Fitting score: {score}")
    
    print(f"Truth Label: {labels[0]}")
    print(f"Predicted Label: {reg.predict(features[0].reshape(1, -1))}")