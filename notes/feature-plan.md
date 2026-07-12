This dataset doesn't include a user ID, so true per-user features (like comparing a transaction to that specific user's own history) aren't possible here. Instead, I adapted the plan to use time-window based features that look at patterns across all transactions in a given time period, as a reasonable substitute. In a real production system with user IDs, per-user velocity and deviation features would be stronger signals than what's used here.

Features planned for Week 2:



Hour of day — converted from the Time column, to check if fraud clusters around unusual hours like late night.

Time since previous transaction — the gap between consecutive transactions, since unusually short gaps can indicate automated or rapid-fire suspicious activity.

Transaction count in rolling time windows — how many transactions happened in the last minute system-wide, as a proxy for sudden bursts of activity (a substitute for per-user velocity).

Log-transformed Amount — since raw transaction amounts are heavily skewed with a few large outliers, applying a log transform makes this feature more stable for the model to learn from.

Amount relative to a rolling average — compares a transaction's amount to the recent average, as a weaker substitute for "deviation from user's own average spending."

Existing V1-V28 columns — already engineered by the original dataset creators using PCA, used directly alongside the new features above.

