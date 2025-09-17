-- ===============================================
-- Queries for Sales & Marketing Dashboard Project
-- ===============================================

-- 1) Total revenue and orders by region
SELECT c.region,
       COUNT(DISTINCT s.order_id) AS total_orders,
       ROUND(SUM(s.revenue), 2) AS total_revenue
FROM sales s
JOIN customers c ON s.customer_id = c.customer_id
GROUP BY c.region
ORDER BY total_revenue DESC;

-- 2) Top 5 customers by total spend
SELECT c.customer_id,
       c.region,
       ROUND(SUM(s.revenue), 2) AS total_spent
FROM sales s
JOIN customers c ON s.customer_id = c.customer_id
GROUP BY c.customer_id, c.region
ORDER BY total_spent DESC
LIMIT 5;

-- 3) Top 20 products by revenue (fixed: removed product_name)
SELECT p.product_id, p.category,
       SUM(s.quantity) AS units_sold,
       ROUND(SUM(s.revenue), 2) AS revenue
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_id, p.category
ORDER BY revenue DESC
LIMIT 20;

-- 4) Average order value (AOV)
SELECT ROUND(SUM(s.revenue) * 1.0 / COUNT(DISTINCT s.order_id), 2) AS avg_order_value
FROM sales s;

-- 5) Customer segmentation by revenue
SELECT c.region,
       COUNT(DISTINCT s.customer_id) AS unique_customers,
       ROUND(SUM(s.revenue), 2) AS revenue
FROM sales s
JOIN customers c ON s.customer_id = c.customer_id
GROUP BY c.region
ORDER BY revenue DESC;

-- 6) Total revenue across all years
SELECT ROUND(SUM(revenue), 2) AS total_revenue
FROM sales;

-- 7) Yearly revenue by category
SELECT strftime('%Y', date) AS year,
       p.category,
       ROUND(SUM(s.revenue), 2) AS revenue
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY year, p.category
ORDER BY year, revenue DESC;

-- 8) Monthly revenue trend (fixed: replaced placeholder)
SELECT strftime('%Y-%m', date) AS month,
       ROUND(SUM(revenue), 2) AS total_revenue
FROM sales
GROUP BY month
ORDER BY month;
