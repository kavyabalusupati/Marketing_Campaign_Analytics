import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = pd.DataFrame({
    "Age": np.random.randint(18, 60, n),
    "Income": np.random.randint(20000, 80000, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "CampaignType": np.random.choice(["Email", "Social", "Ads"], n),
    "PreviousResponse": np.random.choice([0, 1], n),
})

data["Response"] = (
    ((data["Age"] < 40) & (data["Income"] > 40000)) |
    (data["PreviousResponse"] == 1)
).astype(int)

data.to_csv("dataset/marketing_campaign.csv", index=False)

print("✅ 500 rows dataset created!")