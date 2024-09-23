import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import linprog
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('pizza_delivery_data.csv')

data = load_data()

st.title("Pizza Delivery Optimization App")
st.write("""
This app optimizes a pizza deliveries by minimizing delivery costs while maximizing customer satisfaction.
""")

# Sidebar for plot selection
plot_type = st.sidebar.selectbox(
    "Choose the type of plot to display:",
    ("Line Plot", "Scatter Plot", "Box Plot", "Correlation Matrix", "Distribution Plot" )
)

st.header("Data Exploration and Visualization")
# Distribution Plot
if plot_type == "Distribution Plot":
    st.subheader("Distribution Plot")
    dist_var = st.selectbox("Choose a variable for distribution plot:", 
                            options=['delivery_time', 'pizza_quality', 'cost', 'customer_satisfaction'])
    bins = st.slider("Number of bins:", min_value=5, max_value=30, value=10)
    
    # Use Plotly for histogram
    fig = px.histogram(data, x=dist_var, nbins=bins, title=f'Distribution of {dist_var.capitalize()}', marginal="rug", 
                       histnorm='density', opacity=0.75)
    fig.update_layout(xaxis_title=dist_var.capitalize(), yaxis_title='Density', showlegend=False)
    st.plotly_chart(fig)

# Scatter Plot
elif plot_type == "Scatter Plot":
    st.subheader("Scatter Plot")
    x_var = st.selectbox("X-axis variable:", options=['delivery_time', 'distance', 'order_size'])
    y_var = st.selectbox("Y-axis variable:", options=['customer_satisfaction', 'cost'])
    
    # Use Plotly for scatter plot
    fig = px.scatter(data, x=x_var, y=y_var, color='rush_hour', title=f'{x_var.capitalize()} vs {y_var.capitalize()}', 
                     labels={x_var: x_var.capitalize(), y_var: y_var.capitalize()})
    st.plotly_chart(fig)

# Box Plot
elif plot_type == "Box Plot":
    st.subheader("Box Plot")
    y_var = st.selectbox("Choose a variable to compare:", options=['customer_satisfaction', 'cost'])
    
    # Use Plotly for box plot
    fig = px.box(data, x='rush_hour', y=y_var, title=f'{y_var.capitalize()} vs Rush Hour', 
                 labels={'rush_hour': 'Rush Hour', y_var: y_var.capitalize()})
    st.plotly_chart(fig)

# Correlation Matrix
elif plot_type == "Correlation Matrix":
    st.subheader("Correlation Matrix")
    corr_matrix = data.corr()

    # Use Plotly for heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation Coefficient')
    ))
    fig.update_layout(title="Correlation Matrix", xaxis_nticks=36)
    st.plotly_chart(fig)

# Line Plot
elif plot_type == "Line Plot":
    st.subheader("Line Plot")
    time_var = st.selectbox("Select variable to visualize over time:", options=['cost', 'customer_satisfaction'])
    
    # Use Plotly for line plot
    fig = px.line(data, y=time_var, title=f'{time_var.capitalize()} Over Time', labels={'index': 'Time', time_var: time_var.capitalize()})
    st.plotly_chart(fig)

# ---- Predictive Modelling with Model Selection ----
st.header("Predictive Modelling with Adjustable Models")

# Sidebar for model selection
st.sidebar.header("Model Selection for Predictions")
delivery_model_choice = st.sidebar.selectbox("Choose the Model for Delivery Time Prediction", 
                                          options=["Random Forest", "Decision Tree", "Linear Regression"])
satisfaction_model_choice = st.sidebar.selectbox("Choose the Model for Customer Satisfaction Prediction", 
                                                options=["Random Forest", "Decision Tree", "Linear Regression"])

# Sidebar for user input
st.sidebar.header("Prediction Parameters")

# Delivery time prediction parameters
order_size = st.sidebar.slider("Order Size", min_value=1, max_value=10, value=2)
distance = st.sidebar.slider("Distance (miles)", min_value=0, max_value=20, value=5)
rush_hour = st.sidebar.selectbox("Rush Hour", options=[1, 0], index=0)

# Pizza quality prediction parameters
pizza_quality = st.sidebar.slider("Pizza Quality (1-10)", min_value=1, max_value=10, value=8)

# Prepare data for prediction
X_delivery = data[['order_size', 'distance', 'rush_hour']]
y_delivery = data['delivery_time']
X_satisfaction = data[['pizza_quality', 'delivery_time']]
y_satisfaction = data['customer_satisfaction']

# Fit models based on user selection
if delivery_model_choice == "Random Forest":
    delivery_model = RandomForestRegressor(n_estimators=100, max_depth=10)
elif delivery_model_choice == "Decision Tree":
    delivery_model = DecisionTreeRegressor(max_depth=10)
else:
    delivery_model = LinearRegression()

if satisfaction_model_choice == "Random Forest":
    satisfaction_model = RandomForestRegressor(n_estimators=100, max_depth=10)
elif satisfaction_model_choice == "Decision Tree":
    satisfaction_model = DecisionTreeRegressor(max_depth=10)
else:
    satisfaction_model = LinearRegression()

# Train models
delivery_model.fit(X_delivery, y_delivery)
satisfaction_model.fit(X_satisfaction, y_satisfaction)

# Prepare input for predictions
X_delivery_input = np.array([[order_size, distance, rush_hour]])
predicted_delivery_time = delivery_model.predict(X_delivery_input)[0]

X_satisfaction_input = np.array([[pizza_quality, predicted_delivery_time]])
predicted_satisfaction = satisfaction_model.predict(X_satisfaction_input)[0]

st.write(f"**Predicted Delivery Time**: {predicted_delivery_time:.2f} minutes")
st.write(f"**Predicted Customer Satisfaction**: {predicted_satisfaction:.2f}/10")

# ---- Optimization Section ----
st.header("Optimization (Linear Programming)")

st.write("""
Optimize the delivery process by minimizing cost while ensuring customer satisfaction. The predicted values for delivery time and customer satisfaction are used for the optimization.
""")

# Sidebar for cost parameters
st.sidebar.header("Cost Parameters")
delivery_cost = st.sidebar.slider("Delivery Cost per Mile", min_value=0.5, max_value=5.0, value=1.0)
base_cost = st.sidebar.slider("Base Delivery Cost", min_value=5.0, max_value=20.0, value=10.0)
target_satisfaction = st.sidebar.slider("Target Customer Satisfaction (1-10)", min_value=1, max_value=10, value=7)

# Define the optimization problem
# x1: cost to minimize
# x2: customer satisfaction

# Objective: minimize cost (maximize satisfaction)
c = [delivery_cost * distance + base_cost, -predicted_satisfaction]  # Minimize cost, maximize satisfaction
A = [[-1, 0],   # Constraint: cost >= base + distance cost
     [0, 1]]    # Constraint: satisfaction >= target_satisfaction
b = [-base_cost - delivery_cost * distance, target_satisfaction]

# Solve the linear programming problem
result = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (target_satisfaction, predicted_satisfaction)])

# Check if the optimization was successful
if result.success:
    # Extract the optimized values
    optimized_cost = result.x[0]
    optimal_satisfaction = result.x[1]
    
    st.subheader("Optimization Results")
    st.write(f"**Optimized Delivery Cost:** £{optimized_cost:.2f}")
    st.write(f"**Optimized Customer Satisfaction:** {optimal_satisfaction:.2f}/10")
    
    # Visualization of cost vs satisfaction trade-off
    labels = ['Cost', 'Customer Satisfaction']
    sizes = [optimized_cost, optimal_satisfaction]
    colors = ['#ffcc00', '#3366cc']
    
    # Use Plotly for pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, marker_colors=colors, hole=0.3)])
    fig.update_layout(title="Optimized Cost vs Customer Satisfaction")
    st.plotly_chart(fig)
else:
    st.error("Optimization failed. Please check the input values or constraints.")

st.write("""
### Factors Considered:
- **Delivery Time**: The delivery time is predicted based on distance, rush hour, and order size.
- **Pizza Quality**: Pizza quality plays a role in customer satisfaction.
- **Cost Optimization**: The optimization minimizes delivery cost while ensuring customer satisfaction is met.

### Potential Improvements:
- **Dynamic Pricing**: Introduce dynamic pricing based on real-time traffic or demand.
- **Customer Preferences**: Include customer feedback data to improve satisfaction predictions.
- **Advanced Optimization**: Use non-linear optimization for more realistic scenarios.
""")
