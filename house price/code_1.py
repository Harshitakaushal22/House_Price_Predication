import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore", category=UserWarning)
file_path = r"C:\Users\workk\Downloads\data.csv"
df = pd.read_csv(file_path)
print("Columns in dataset:")
print(df.columns.tolist())
numeric_features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors']
categorical_features = ['city', 'waterfront', 'condition', 'view']
target_column = 'price'
all_cols = numeric_features + categorical_features + [target_column]
df = df.dropna(subset=all_cols)
X = df[numeric_features + categorical_features]
y = df[target_column]
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
model = pipeline.named_steps['regressor']
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
coefficients = pd.Series(model.coef_, index=feature_names).sort_values(key=abs, ascending=False)
print("\nTop Influential Features:")
print(coefficients.head(10))
