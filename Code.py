
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Step 2: Generating a sample dataset...")
data = {
    'YearsExperience': np.random.randint(0, 25, size=500),
    'EducationLevel': np.random.choice(['Bachelors', 'Masters', 'PhD'], size=500, p=[0.5, 0.3, 0.2]),
    'JobRole': np.random.choice(['Developer', 'Data Scientist', 'Manager', 'Analyst'], size=500, p=[0.4, 0.2, 0.2, 0.2])
}
df = pd.DataFrame(data)


df['Salary'] = 50000 + (df['YearsExperience'] * 2500) + \
               df['EducationLevel'].map({'Bachelors': 10000, 'Masters': 25000, 'PhD': 40000}) + \
               df['JobRole'].map({'Developer': 15000, 'Data Scientist': 30000, 'Manager': 40000, 'Analyst': 5000}) + \
               np.random.randint(-5000, 5000, size=500) # Some random noise

print("Sample dataset created successfully.")
print("First 5 rows of the dataset:")
print(df.head())
print("-" * 30)


print("Step 3: Preprocessing data...")

X = df.drop('Salary', axis=1)
y = df['Salary']


categorical_features = ['EducationLevel', 'JobRole']


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' 
)
print("Preprocessing pipeline created.")
print("-" * 30)


print("Step 4: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("-" * 30)

print("Step 5: Training the RandomForestRegressor model...")

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)



print("Step 6: Evaluating the model...")

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Results:")
print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"  R-squared (R²): {r2:.4f}")
print("-" * 30)


print("Step 7: Creating a visualization of actual vs. predicted salaries...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2) # Perfect prediction line
plt.title('Actual vs. Predicted Salaries', fontsize=16)
plt.xlabel('Actual Salary ($)', fontsize=12)
plt.ylabel('Predicted Salary ($)', fontsize=12)
plt.grid(True)
plt.show()
print("Visualization created.")
print("-" * 30)


print("Step 8: Making a prediction on a new data sample...")

new_employee_data = pd.DataFrame({
    'YearsExperience': [10],
    'EducationLevel': ['Masters'],
    'JobRole': ['Data Scientist']
})


predicted_salary = model.predict(new_employee_data)

print(f"Data for new employee: \n{new_employee_data}")
print(f"\nPredicted Salary: ${predicted_salary[0]:,.2f}")
print("=" * 50)
print("CODE EXECUTION FINISHED")
print("=" * 50)
