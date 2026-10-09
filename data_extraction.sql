sql
-- SQL Extraction Script for Sales Forecasting Dataset
-- Target: Aggregate daily transactions to build modeling features
SELECT 
    CAST(transaction_date AS DATE) as date,
    product_category,
    store_id,
    SUM(sale_amount) as sales,
    COUNT(distinct transaction_id) as total_transactions
FROM historical_sales_db.orders
WHERE transaction_date BETWEEN '2023-01-01' AND '2025-12-31'
GROUP BY CAST(transaction_date AS DATE), product_category, store_id
ORDER BY date ASC;
