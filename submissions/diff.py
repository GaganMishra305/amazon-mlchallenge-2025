import pandas as pd

sub1 = pd.read_csv("fifth51.csv")
sub2 = pd.read_csv("current_submission.csv")

diff = sub2["price"] - sub1["price"]

print("Mean difference:", diff.mean())
print("Std deviation of difference:", diff.std())
print("Correlation between old and new:", sub1["price"].corr(sub2["price"]))
