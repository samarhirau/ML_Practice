import seaborn as sns
import matplotlib.pyplot as plt


data = {
    "sex": ["male", "female", "female", "male", "female", "male", "female", "male", "female"],
    "age": [23, 21, 22, 24, 20, 25, 23, 22, 21],
    "height": [175, 160, 165, 180, 155, 170, 168, 182, 158],
    "weight": [70, 55, 60, 80, 50, 75, 65, 85, 52]
}

# set the style of the plot
sns.set_style(style='whitegrid')

# quick view of plot
sns.scatterplot(data=data, x="height", y="weight", hue="sex", s=100)
plt.show()
