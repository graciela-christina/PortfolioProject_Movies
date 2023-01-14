SELECT * FROM movies_cleaned

SELECT TOP 10 name, country, gross, budget, votes, year
FROM movies_cleaned
ORDER BY gross DESC

SELECT TOP 10 name, country, gross, budget, votes, year
FROM movies_cleaned
ORDER BY budget DESC

SELECT country, SUM(budget) AS total_budget, SUM(gross) AS total_gross_revenue, SUM(votes) AS total_votes
FROM movies_cleaned
WHERE year = '2019'
GROUP BY country
ORDER BY total_gross_revenue DESC