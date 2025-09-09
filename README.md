
# 🛍️ Retail Sales Forecasting with Gradient Boosting

## 📌 Project Overview
This project builds an **end-to-end machine learning pipeline** to **forecast retail product sales** using **Python** and **Gradient Boosting Regressor (GBR)**.  

By predicting **total sales per transaction** and identifying **key business drivers**, this solution supports **data-driven decision-making** for:
- Inventory management
- Pricing optimization
- Marketing budget allocation

---

## 🎯 Key Objectives
- Develop a **scalable ML pipeline** for retail forecasting.
- Handle **data preprocessing**, including:
  - Missing value imputation
  - Scaling of numeric features
  - Encoding categorical variables
- Apply **Gradient Boosting** to improve prediction accuracy.
- Tune hyperparameters using **GridSearchCV**.
- Visualize results and **extract business insights**.

---

## 🛠️ Tech Stack
| Category            | Tools & Libraries |
|---------------------|-------------------|
| **Programming**     | Python |
| **ML & Modeling**   | Scikit-learn, GradientBoostingRegressor |
| **Data Handling**   | Pandas, NumPy |
| **Visualization**   | Matplotlib |
| **Workflow**        | Pipelines, GridSearchCV |
| **Deployment Ready**| Docker, AWS SageMaker (future scope) |

---

## 📂 Project Structure


## 📊 Dataset
Since no public dataset was provided, a **synthetic retail dataset** was generated to simulate real-world transactions.

| Feature         | Description |
|----------------|-------------|
| **InvoiceNo**  | Unique invoice identifier |
| **StockCode**  | Product code |
| **Description**| Product description |
| **Quantity**   | Units sold per transaction |
| **UnitPrice**  | Price per unit |
| **InvoiceDate**| Date and time of transaction |
| **Country**    | Transaction location |
| **MarketingSpend** | Marketing budget used for that period |
| **Target (TotalSales)** | `Quantity × UnitPrice` |

---

## 🚀 How to Run the Project

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/retail-sales-forecasting.git
cd retail-sales-forecasting
