#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("AirQualityUCI_with_quality_24h.csv")

# 2. Replace -200 with NaN
df = df.replace(-200, np.nan)

# 3. Create Datetime column and set as index
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], 
    format="%m/%d/%Y %H:%M:%S"
)

df = df.drop(columns=["Date", "Time"])
df = df.set_index("Datetime")

# 4. Drop NMHC(GT) (90% missing)
df = df.drop(columns=["NMHC(GT)"])

# 5. Interpolate only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].interpolate(method="time")

# 6. Drop remaining NaNs (if any)
df = df.dropna()

# 7. Check final structure
print(df.info())
print(df.head())


# In[38]:


print(df.dtypes)


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns
#A line plot is the most effective way to visualize how a variable changes over time
sns.set_style("whitegrid")

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(15, 6))

# Plot the CO(GT) data
ax.plot(df['CO(GT)'], label='CO(GT) Concentration', color='blue')

# Set titles and labels
ax.set_title('CO(GT) Trends Over Time', fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("CO(GT) Concentration", fontsize=12)
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[28]:


#A. Time Series Trends
import matplotlib.pyplot as plt

# Plot major pollutants over time
pollutants = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']

plt.figure(figsize=(15, 10))
for i, col in enumerate(pollutants, 1):
    plt.subplot(len(pollutants), 1, i)
    df[col].plot()
    plt.title(f"{col} over time")
    plt.ylabel(col)
    plt.xlabel("Time")
plt.tight_layout()
plt.show()


# In[22]:


# Resample the data to get monthly averages
monthly_avg = df['CO(GT)'].resample('M').mean()

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(15, 6))

# Plot the monthly average CO(GT)
ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-', color='purple')

# Set titles and labels
ax.set_title('Monthly Average of CO(GT) (Seasonal Trend)', fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Average CO(GT) Concentration", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[29]:


#Meteorological Feature Trends
weather = ['T', 'RH', 'AH']

plt.figure(figsize=(15, 7))
for i, col in enumerate(weather, 1):
    plt.subplot(len(weather), 1, i)
    df[col].plot(color='teal')
    plt.title(f"{col} over time")
    plt.ylabel(col)
plt.tight_layout()
plt.show()


# In[41]:


# Create box plots for key numerical variables
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Box plot for CO(GT)
sns.boxplot(y=df['CO(GT)'], ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of CO(GT) with Anomalies')
axes[0].set_ylabel('CO(GT) Concentration')

# Box plot for Temperature (T)
sns.boxplot(y=df['T'], ax=axes[1], color='lightgreen')
axes[1].set_title('Distribution of Temperature (T) with Anomalies')
axes[1].set_ylabel('Temperature (°C)')

plt.tight_layout()
plt.show()


# In[43]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Select the input features and the target variable
features = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
            'PT08.S5(O3)', 'T', 'RH', 'AH']
target = ['CO(GT)']

# Select the data and scale it
data = df[features].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for the LSTM model
def create_sequences(input_data, output_data, time_steps):
    X, y = [], []
    for i in range(len(input_data) - time_steps - 1):
        X.append(input_data[i:(i + time_steps)])
        y.append(output_data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 60
X_seq, y_seq = create_sequences(scaled_data, scaled_data[:, features.index('CO(GT)')], time_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (this will produce the loss output)
print("Training the model...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Make predictions on the test set
print("\nMaking predictions and evaluating the model...")
y_pred = model.predict(X_test)

# Inverse transform the predictions and true values to the original scale
dummy_array = np.zeros((len(X_test), len(features)))
dummy_array[:, features.index('CO(GT)')] = y_test
y_test_original = scaler.inverse_transform(dummy_array)[:, features.index('CO(GT)')]

dummy_array = np.zeros((len(X_test), len(features)))
dummy_array[:, features.index('CO(GT)')] = y_pred.flatten()
y_pred_original = scaler.inverse_transform(dummy_array)[:, features.index('CO(GT)')]

# Calculate the evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test_original, y_pred_original)


# Print the results
print("\nModel Evaluation Results:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R^2): {r2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# LSTM (Univariate) — standard baseline for single-series forecasting; shows how much the pollutant’s own history explains future NO₂. Easier to interpret and fast to train on a single channel.
# 1. Prepare the data
# -----------------------
series = df['NO2(GT)'].values.reshape(-1,1)

# Scale between 0 and 1
scaler_uni = MinMaxScaler()
series_scaled = scaler_uni.fit_transform(series)

# Create sequences
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 60
X_u, y_u = create_sequences(series_scaled, seq_len=seq_len)

# Train-test split (time-based, 80/20)
split = int(len(X_u) * 0.8)
X_train_u, X_test_u = X_u[:split], X_u[split:]
y_train_u, y_test_u = y_u[:split], y_u[split:]

# -----------------------
# 2. Build & Train LSTM
# -----------------------
model_u = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model_u.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("Training Univariate LSTM...")
history_u = model_u.fit(
    X_train_u, y_train_u,
    validation_data=(X_test_u, y_test_u),
    epochs=100,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# -----------------------
# 3. Predictions
# -----------------------
y_pred_u = model_u.predict(X_test_u)

# Inverse transform
dummy = np.zeros((len(y_pred_u), 1))
dummy[:,0] = y_pred_u.flatten()
y_pred_u_orig = scaler_uni.inverse_transform(dummy)[:,0]

dummy[:,0] = y_test_u.flatten()
y_test_u_orig = scaler_uni.inverse_transform(dummy)[:,0]

# -----------------------
# 4. Evaluation
# -----------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

rmse = np.sqrt(mean_squared_error(y_test_u_orig, y_pred_u_orig))
mae = mean_absolute_error(y_test_u_orig, y_pred_u_orig)
r2 = r2_score(y_test_u_orig, y_pred_u_orig)
mape = mean_absolute_percentage_error(y_test_u_orig, y_pred_u_orig)

print("\nUnivariate LSTM Results:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
print(f"MAPE: {mape:.2f}%")

# -----------------------
# 5. Plot Results
# -----------------------
plt.figure(figsize=(12,5))
plt.plot(y_test_u_orig[:200], label="Actual", color="black")
plt.plot(y_pred_u_orig[:200], label="Predicted", color="teal")
plt.title("Univariate LSTM - NO₂ Prediction")
plt.xlabel("Time (test samples)")
plt.ylabel("NO₂ concentration")
plt.legend()
plt.show()


# In[47]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.model_selection import train_test_split

 #GRU (Multivariate) — GRU is computationally lighter than LSTM yet powerful; multivariate input lets the model learn cross-variable interactions (pollutants + weather). Different cell type + different input dimensionality = clearly distinct models.
# 1. Data Preparation
# Data preparation
# =========================
features = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
            'PT08.S5(O3)', 'T', 'RH', 'AH']
target = 'NO2(GT)'

data = df[features].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(input_data, output_data, time_steps):
    X, y = [], []
    for i in range(len(input_data) - time_steps - 1):
        X.append(input_data[i:(i + time_steps)])
        y.append(output_data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 60
X_seq, y_seq = create_sequences(scaled_data, scaled_data[:, features.index(target)], time_steps)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

# =========================
# GRU Model
# =========================
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training Multivariate GRU...")
history = model.fit(
    X_train, y_train, epochs=30, batch_size=64,
    validation_data=(X_test, y_test), verbose=1
)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test)

# Inverse transform: put predictions back into original scale
dummy_array = np.zeros((len(y_test), len(features)))
dummy_array[:, features.index(target)] = y_test
y_test_original = scaler.inverse_transform(dummy_array)[:, features.index(target)]

dummy_array = np.zeros((len(y_pred), len(features)))
dummy_array[:, features.index(target)] = y_pred.flatten()
y_pred_original = scaler.inverse_transform(dummy_array)[:, features.index(target)]

# =========================
# Evaluation
# =========================
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

print("\nMultivariate GRU Results (Inverse Transformed):")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
print(f"MAPE: {mape:.2f}%")

# =========================
# Plot Results
# =========================
plt.figure(figsize=(12, 6))
plt.plot(y_test_original[:200], label="True NO₂", color="blue")
plt.plot(y_pred_original[:200], label="Predicted NO₂", color="red")
plt.title("Multivariate GRU Predictions vs True Values")
plt.xlabel("Time Steps")
plt.ylabel("NO₂ Concentration")
plt.legend()
plt.show()


# In[48]:


# Results from your two models
results = {
    "Model": ["Univariate LSTM", "Multivariate GRU"],
    "RMSE": [22.18, 15.78],
    "MAE": [16.28, 11.51],
    "R²": [0.83, 0.88],
    "MAPE (%)": [12.74, 12.70]
}

# Create DataFrame
comparison_df = pd.DataFrame(results)

print("\nModel Comparison Table:")
print(comparison_df)

# Optional: display nicely if running in Jupyter/Colab
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
sns.heatmap(comparison_df.set_index("Model"), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Model Performance Comparison")
plt.show()


# In[49]:


#From the chart, it appears that the Multivariate GRU model slightly outperforms the Univariate LSTM in all four metrics, especially in RMSE, MAE, and R², where Multivariate GRU has lower error values and a higher R².
#In conclusion, the Multivariate GRU model is better suited for this problem because it leverages the full context of air quality and weather data, leading to more robust and accurate predictions of NO₂ levels.


# In[52]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

# --------------------------
# 1. Prepare Data
# --------------------------
# Target pollutant
target_col = "NO2(GT)"  

# Features for multivariate
features = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
            'PT08.S5(O3)', 'T', 'RH', 'AH']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features].values)

# Function to create sequences
def create_sequences(input_data, output_data, time_steps=60):
    X, y = [], []
    for i in range(len(input_data) - time_steps):
        X.append(input_data[i:(i + time_steps)])
        y.append(output_data[i + time_steps])
    return np.array(X), np.array(y)

# Univariate (only NO2(GT))
no2_scaled = scaler.fit_transform(df[[target_col]])
X_uni, y_uni = create_sequences(no2_scaled, no2_scaled[:, 0], time_steps=60)

# Multivariate (all features → predict NO2(GT))
X_multi, y_multi = create_sequences(scaled_data, scaled_data[:, features.index(target_col)], time_steps=60)

# --------------------------
# 2. Define Models
# --------------------------
def build_univariate_lstm(input_shape, units=50, lr=0.001):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

def build_multivariate_gru(input_shape, units=50, lr=0.001):
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
    model.add(GRU(units=units))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

# --------------------------
# 3. Cross-Validation Function
# --------------------------
def cross_validate(X, y, build_fn, units, lr, n_splits=3, epochs=5, batch_size=64):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_fn((X.shape[1], X.shape[2]), units=units, lr=lr)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        val_losses.append(history.history["val_loss"][-1])

    return np.mean(val_losses), np.std(val_losses)

# --------------------------
# 4. Hyperparameter Grid Search
# --------------------------
param_grid = [
    {"units": 32, "lr": 0.001},
    {"units": 64, "lr": 0.001},
    {"units": 64, "lr": 0.0005},
]

print(" Univariate LSTM Cross-Validation & Hyperparameter Tuning")
for params in param_grid:
    mean_loss, std_loss = cross_validate(X_uni, y_uni, build_univariate_lstm,
                                         units=params["units"], lr=params["lr"])
    print(f"Units={params['units']}, LR={params['lr']} → Loss={mean_loss:.4f} ± {std_loss:.4f}")

print("\n Multivariate GRU Cross-Validation & Hyperparameter Tuning")
for params in param_grid:
    mean_loss, std_loss = cross_validate(X_multi, y_multi, build_multivariate_gru,
                                         units=params["units"], lr=params["lr"])
    print(f"Units={params['units']}, LR={params['lr']} → Loss={mean_loss:.4f} ± {std_loss:.4f}")


# In[53]:


#Overall, the Multivariate GRU outperformed the Univariate LSTM during cross-validation, confirming that using multiple features improves predictive accuracy.


# In[82]:


# PCMCI is used to identify lagged causal relationships in the time-series data, capturing temporal dependencies rigorously.
# Correlation-based edges are used to infer contemporaneous relationships, replacing NOTEARS/CDT due to installation issues.
# Combining these two complementary methods allows us to capture both temporal and contemporaneous effects,
# providing robust insights for predictive modeling and satisfying the requirement for two causal discovery methods.


# In[56]:


from sklearn.preprocessing import StandardScaler

# choose variables (drop categorical label)
vars_to_use = ['CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)',
               'PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)',
               'PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']

# 1) optionally remove diurnal cycle (detrend) to help stationarity
df_sub = df[vars_to_use].copy()
hour_means = df_sub.groupby(df_sub.index.hour).transform('mean')
df_detrended = df_sub - hour_means  # removes hour-of-day seasonal mean

# 2) fill small remaining NaNs (forward/backfill)
df_detrended = df_detrended.fillna(method='ffill').fillna(method='bfill')

# 3) standardize
scaler = StandardScaler()
data_std = pd.DataFrame(scaler.fit_transform(df_detrended), columns=vars_to_use, index=df_detrended.index)



# In[62]:


from tigramite.data_processing import DataFrame as tgDataFrame
from tigramite.pcmci import PCMCI
import networkx as nx
from tigramite.independence_tests.parcorr import ParCorr


# data_std from preprocessing (T x N)
data_arr = data_std.values  # shape (T, N)
tg_df = tgDataFrame(data_arr, var_names=list(data_std.columns))

pcmci = PCMCI(dataframe=tg_df, cond_ind_test=ParCorr(), verbosity=1)

# choose max lag: justify in report (e.g., tau_max=6 for 6 hours; try 24 for 1-day)
tau_max = 6
results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.05)

# p_matrix and val_matrix contain p-values and test statistics
p_matrix = results['p_matrix']      # shape (N, N, tau_max+1)
val_matrix = results['val_matrix']  # test statistic or partial correlation

# collect significant lagged links (j -> i with lag)
alpha = 0.05
links = []
N = len(data_std.columns)
names = list(data_std.columns)
for j in range(N):
    for i in range(N):
        for lag in range(1, tau_max+1):   # lag>0 are temporal
            pval = p_matrix[j, i, lag]
            if (pval is not None) and (pval <= alpha):
                strength = val_matrix[j, i, lag]
                links.append((names[j], f"t-{lag}", "->", names[i], "t", f"p={pval:.3f}", f"str={strength:.3f}"))

# Print top found links
for l in sorted(links, key=lambda x: float(x[-1].split('=')[1]), reverse=True)[:50]:
    print(l)

# Optional: build a lagged network visualization (collapse lags or show top-edges)
G = nx.DiGraph()
for var in names: G.add_node(var)
for src, s_lag, arrow, tgt, t_lag, p, strg in links:
    # collapse using src (use only src->tgt ignoring lag in node labels if you prefer)
    G.add_edge(src, tgt, weight=float(strg.split('=')[1]), pval=float(p.split('=')[1]))

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=2)
nx.draw(G, pos, with_labels=True, node_size=800, font_size=9)
edge_labels = { (u,v): f"{d['weight']:.2f}" for u,v,d in G.edges(data=True) }
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.title("PCMCI (lagged) inferred graph (collapsed edges)")
plt.show()


# In[63]:


#Temperature (T) and humidity (RH, AH) appear to have a strong influence on many of the sensors, possibly due to their impact on the physical or chemical properties of the air.

#NO2 and CO play a central role in influencing various sensor readings, possibly due to their association with pollutants in the air.

#The influence between sensor readings suggests a potential calibration or cross-sensitivity issue, which is common in air quality sensors that can respond to multiple pollutants.


# In[81]:


# ---------------------------
# Time-Series Causal Discovery (CDT-free, Python 3.11 compatible)
# ---------------------------

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------
# 1. Assume `data_std` is your standardized DataFrame (T x N)
# ---------------------------

L = 3  # number of past lags to include
vars_list = list(data_std.columns)

# ---------------------------
# 2. Build lagged features
# ---------------------------

lagged_arrays = []
colnames = []

for lag in range(L+1):
    lagged = data_std.shift(lag).iloc[L:]  # drop first L rows
    lagged_arrays.append(lagged.values)
    colnames += [f"{v}_lag{lag}" for v in vars_list]

X_lagged = np.hstack(lagged_arrays)
data_lagged = pd.DataFrame(X_lagged, columns=colnames)

# ---------------------------
# 3. Compute pairwise correlations as a proxy for causal links
# ---------------------------

corr_matrix = data_lagged.corr().abs()  # absolute correlations

# ---------------------------
# 4. Threshold correlations to define edges
# ---------------------------

threshold = 0.3  # tune as needed
edges_interesting = []

for i, tgt in enumerate(colnames):
    if tgt.endswith("_lag0"):  # focus on current variables
        for j, src in enumerate(colnames):
            if i != j and corr_matrix.iloc[i, j] > threshold:
                edges_interesting.append((src, tgt, corr_matrix.iloc[i, j]))

# ---------------------------
# 5. Collapse lagged features to original variables for visualization
# ---------------------------

G = nx.DiGraph()
for v in vars_list:
    G.add_node(v)

for s, t, w in edges_interesting:
    src_var = s.split("_lag")[0]
    tgt_var = t.split("_lag")[0]
    if src_var != tgt_var:
        G.add_edge(src_var, tgt_var, weight=w)

# ---------------------------
# 6. Visualize DAG
# ---------------------------

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=1)
nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue', arrowsize=20)
edge_labels = { (u,v): f"{d['weight']:.2f}" for u,v,d in G.edges(data=True) }
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.title("Causal DAG (correlation-based on lagged features)")
plt.show()

# ---------------------------
# 7. Print top causal edges
# ---------------------------

edges_interesting_sorted = sorted(edges_interesting, key=lambda x: x[2], reverse=True)
print("Top causal edges:")
for s, t, w in edges_interesting_sorted[:20]:
    print(f"{s} -> {t}  (weight={w:.2f})")


# In[86]:


# ---------------------------
# Step 6 alternative: SCM + causal effect estimation
# ---------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import networkx as nx
import matplotlib.pyplot as plt

# Assume G is your combined DAG from Step 5 (PCMCI + correlation edges)

# ---------------------------
# 1. Build SCM: regress each variable on its parents
# ---------------------------
scm = {}
for tgt in data_std.columns:
    parents = [src for src, child in G.edges() if child == tgt]
    if parents:
        X_parents = data_std[parents]
        y = data_std[tgt]
        model = LinearRegression().fit(X_parents, y)
        scm[tgt] = (parents, model)
    else:
        scm[tgt] = ([], None)

# ---------------------------
# 2. Simulate interventions using SCM regression
# ---------------------------
def intervene(scm, data, target_var, value):
    data_sim = data.copy()
    data_sim[target_var] = value
    # propagate downstream effects
    downstream = [child for child in data.columns if target_var in scm[child][0]]
    for child in downstream:
        parents, model = scm[child]
        if model:
            X_parents = data_sim[parents]
            data_sim[child] = model.predict(X_parents)
    return data_sim

# Example: intervene on Temperature
T_values = [10, 20, 30]
for val in T_values:
    sim_data = intervene(scm, data_std, 'T', val)
    print(f"Mean C6H6(GT) under do(T={val}): {sim_data['C6H6(GT)'].mean():.3f}")

# ---------------------------
# 3. Compare SCM regression coefficients (causal effects) with DAG edges
# ---------------------------
print("\nSCM regression coefficients (effect sizes):")
for tgt, (parents, model) in scm.items():
    if model:
        for i, p in enumerate(parents):
            print(f"{p} -> {tgt}: {model.coef_[i]:.3f}")

# ---------------------------
# 4. Optional: visualize SCM DAG with effect sizes
# ---------------------------
plt.figure(figsize=(12,8))
pos = nx.spring_layout(G, seed=1)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
nx.draw_networkx_labels(G, pos, font_size=10)

# Draw edges with SCM regression coefficients
edge_labels = {}
for src, tgt in G.edges():
    if tgt in scm and scm[tgt][1] is not None and src in scm[tgt][0]:
        idx = scm[tgt][0].index(src)
        weight = scm[tgt][1].coef_[idx]
        label = f"{weight:.2f}"
    else:
        label = f"{G.edges[src, tgt].get('weight',1.0):.2f}"
    edge_labels[(src, tgt)] = label

nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='black')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("SCM DAG with Edge Effect Sizes (without DoWhy)")
plt.axis('off')
plt.show()


# In[88]:


# ---------------------------
# : Top causal effects table + DAG visualization
# ---------------------------
#  Generate top causal effects table
effects = []

for tgt, (parents, model) in scm.items():
    if model:
        for i, src in enumerate(parents):
            coef = model.coef_[i]
            # Optional: simulate a standardized intervention (1 std above mean)
            sim_data = intervene(scm, data_std, src, 1.0)
            delta = sim_data[tgt].mean() - data_std[tgt].mean()
            
            effects.append({
                "Source": src,
                "Target": tgt,
                "SCM_coef": coef,
                "Intervention_effect": delta
            })

effects_df = pd.DataFrame(effects)
effects_df["abs_coef"] = effects_df["SCM_coef"].abs()
effects_df = effects_df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

print("Top 20 Causal Effects Table:")
display(effects_df.head(20))

#  Visualize top 20 causal effects on DAG
plt.figure(figsize=(12,8))
pos = nx.spring_layout(G, seed=1)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
nx.draw_networkx_labels(G, pos, font_size=10)

# Draw edges for top 20 effects
top_effects = effects_df.head(20)

for _, row in top_effects.iterrows():
    src = row["Source"]
    tgt = row["Target"]
    coef = row["SCM_coef"]
    
    # Edge color by sign, thickness by magnitude
    color = 'green' if coef > 0 else 'red'
    width = abs(coef)*3  # scale for visibility
    nx.draw_networkx_edges(G, pos, edgelist=[(src, tgt)], width=width, edge_color=color, arrowstyle='-|>', arrowsize=20)
    
# Edge labels show coefficient values
edge_labels = {(row["Source"], row["Target"]): f"{row['SCM_coef']:.2f}" for _, row in top_effects.iterrows()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Top 20 Causal Effects (SCM) Highlighted")
plt.axis('off')
plt.show()


# In[89]:


# ---------------------------
#: Combined DAG (PCMCI + SCM top effects)
# ---------------------------

plt.figure(figsize=(14,10))
pos = nx.spring_layout(G, seed=1)  # use previous layout

#  Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
nx.draw_networkx_labels(G, pos, font_size=10)

#  Draw PCMCI lagged edges (temporal)
for u, v, d in G.edges(data=True):
    # thinner, gray edges for temporal (PCMCI) relationships
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1, edge_color='gray', style='dashed', alpha=0.5)

#  Draw top SCM edges (contemporaneous)
top_effects = effects_df.head(20)
for _, row in top_effects.iterrows():
    src = row["Source"]
    tgt = row["Target"]
    coef = row["SCM_coef"]
    
    color = 'green' if coef > 0 else 'red'
    width = abs(coef)*3
    nx.draw_networkx_edges(G, pos, edgelist=[(src, tgt)], width=width, edge_color=color, arrowstyle='-|>', arrowsize=20)

#  Edge labels for top SCM effects
edge_labels = {(row["Source"], row["Target"]): f"{row['SCM_coef']:.2f}" for _, row in top_effects.iterrows()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Combined Causal DAG: Temporal (PCMCI) + Contemporaneous (SCM) Effects")
plt.axis('off')
plt.show()


# In[90]:


# The combined DAG visualizes the causal relationships in the Air Quality dataset.
# Nodes represent variables, gray dashed edges show temporal lagged effects 
# detected by PCMCI (capturing autocorrelation and lagged causality), while 
# colored edges represent the strongest contemporaneous causal effects estimated 
# via the Structural Causal Model (SCM). Edge thickness is proportional to the 
# magnitude of the SCM regression coefficient, and color indicates direction 
# (green = positive effect, red = negative effect). This figure highlights both 
# temporal and contemporaneous drivers of pollutants and meteorological variables, 
# providing a comprehensive overview of the causal structure.


# In[94]:


# -----------------------------------------------------------------------------
#: Final Predictive Model using PCMCI + SCM causal edges
# -----------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# 0. Create lagged features for PCMCI edges
# ---------------------------
L = 1  # maximum lag used in PCMCI edges
lagged_data = data_std.copy()

# Create lagged columns _lag1
for col in data_std.columns:
    for lag in range(1, L+1):
        lagged_data[f"{col}_lag{lag}"] = data_std[col].shift(lag)

# Drop first L rows with NaNs
lagged_data = lagged_data.iloc[L:].reset_index(drop=True)

# ---------------------------
# 1. Define targets and PCMCI + SCM parents
# ---------------------------
targets = ["C6H6(GT)_lag0", "CO(GT)_lag0", "NOx(GT)_lag0"]

# Create lag0 columns for the target variables
for t in ["C6H6(GT)", "CO(GT)", "NOx(GT)"]:
    lagged_data[f"{t}_lag0"] = lagged_data[t]

pcmci_edges = {
    "C6H6(GT)_lag0": ["AH_lag1", "T_lag1", "NOx(GT)_lag1"],
    "CO(GT)_lag0": ["C6H6(GT)_lag1", "NOx(GT)_lag1", "PT08.S1(CO)_lag1"],
    "NOx(GT)_lag0": ["CO(GT)_lag1", "C6H6(GT)_lag1", "NO2(GT)_lag1"]
}

scm_parents = {
    "C6H6(GT)_lag0": ["PT08.S2(NMHC)_lag0", "PT08.S3(NOx)_lag0"],
    "CO(GT)_lag0": ["C6H6(GT)_lag0", "NOx(GT)_lag0", "PT08.S1(CO)_lag0"],
    "NOx(GT)_lag0": ["C6H6(GT)_lag0", "CO(GT)_lag0", "NO2(GT)_lag0"]
}

# ---------------------------
# 2. Ensure all SCM parents exist in lagged_data
# ---------------------------
scm_variables = set()
for parents in scm_parents.values():
    scm_variables.update(parents)

for var in scm_variables:
    if var not in lagged_data.columns:
        orig_var = var.replace("_lag0", "")
        lagged_data[var] = lagged_data[orig_var]

# ---------------------------
# 3. Combine all features
# ---------------------------
all_features = set()
for t in targets:
    all_features.update(pcmci_edges[t])
    all_features.update(scm_parents[t])
all_features = list(all_features)

# ---------------------------
# 4. Prepare feature matrix X and target matrix Y
# ---------------------------
X = lagged_data[all_features]
Y = lagged_data[targets]

# ---------------------------
# 5. Train/test split
# ---------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---------------------------
# 6. Train multi-output Random Forest
# ---------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
multi_rf = MultiOutputRegressor(rf)
multi_rf.fit(X_train, Y_train)

# ---------------------------
# 7. Predict and evaluate
# ---------------------------
Y_pred = multi_rf.predict(X_test)

for i, target in enumerate(targets):
    rmse = mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i], squared=False)
    r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
    print(f"{target}: RMSE={rmse:.4f}, R2={r2:.4f}")


# In[95]:


#The final predictive model integrates causality into machine learning by combining 
# temporal causal relationships discovered with PCMCI (lagged edges) and 
# contemporaneous causal dependencies obtained from the Structural Causal Model (SCM). 
# By using only the parents of each target variable identified from these causal methods 
# as features, the model focuses on the most relevant predictors and reduces spurious 
# correlations. Multi-output Random Forest is then trained to predict multiple pollutants 
# simultaneously, ensuring that both temporal and instantaneous causal influences are 
# respected in the predictions. This approach leverages domain-informed causal structure 
# to improve interpretability and potentially the generalization of the ML model.


# In[ ]:





# In[ ]:





# In[ ]:




