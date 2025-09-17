
-- Sales & Marketing useful queries (SQLite)

-- 1) Monthly revenue & orders
SELECT strftime('%Y-%m', date) AS month,
       COUNT(transaction_id) AS orders,
       ROUND(SUM(revenue), 2) AS revenue
FROM sales
GROUP BY month
ORDER BY month;

-- 2) Revenue by region
SELECT c.region,
       COUNT(s.transaction_id) AS orders,
       ROUND(SUM(s.revenue), 2) AS revenue
FROM sales s
JOIN customers c ON s.customer_id = c.customer_id
GROUP BY c.region
ORDER BY revenue DESC;

-- 3) Top 20 products by revenue
SELECT p.product_id, p.product_name, p.category,
       SUM(s.quantity) AS units_sold,
       ROUND(SUM(s.revenue), 2) AS revenue
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_id
ORDER BY revenue DESC
LIMIT 20;

-- 4) Revenue by category
SELECT category, COUNT(transaction_id) AS orders, ROUND(SUM(revenue),2) AS revenue
FROM sales
GROUP BY category
ORDER BY revenue DESC;

-- 5) Customer RFM (recency = days since last order)
WITH params AS (SELECT MAX(date) AS max_date FROM sales),
rfm AS (
  SELECT customer_id,
         MAX(date) AS last_order,
         COUNT(transaction_id) AS frequency,
         ROUND(SUM(revenue),2) AS monetary
  FROM sales
  GROUP BY customer_id
)
SELECT r.customer_id,
       CAST(JULIANDAY((SELECT max_date FROM params)) - JULIANDAY(r.last_order) AS INTEGER) AS recency,
       r.frequency,
       r.monetary
FROM rfm r
ORDER BY r.monetary DESC
LIMIT 200;

-- 6) Average Order Value (AOV) and Repeat Rate
SELECT ROUND(SUM(revenue)/COUNT(DISTINCT transaction_id),2) AS aov,
       (SELECT ROUND(100.0 * SUM(CASE WHEN cnt > 1 THEN 1 ELSE 0 END) / COUNT(*),2)
        FROM (SELECT customer_id, COUNT(*) AS cnt FROM sales GROUP BY customer_id)) AS repeat_rate_pct
FROM sales;

-- 7) Simple cohort analysis (first_purchase_month vs order_month)
WITH first_purchase AS (
  SELECT customer_id, MIN(strftime('%Y-%m', date)) AS cohort_month
  FROM sales
  GROUP BY customer_id
)
SELECT fp.cohort_month AS cohort,
       strftime('%Y-%m', s.date) AS order_month,
       COUNT(DISTINCT s.customer_id) AS active_customers
FROM sales s
JOIN first_purchase fp ON s.customer_id = fp.customer_id
GROUP BY fp.cohort_month, strftime('%Y-%m', s.date)
ORDER BY fp.cohort_month, order_month;
SQL
