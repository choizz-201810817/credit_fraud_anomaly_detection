import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data under sampling func
def underSample(df, targetCol=''):
    df = df.sample(frac=1) # frac=1은 100% 다 가져오는 것(1로 하면 df의 순서만 바뀜)

    fraudDf = df[df[targetCol]==1]
    num = len(df[df[targetCol]==1])
    
    nonFraudDf = df[df[targetCol]==0][:num]

    normDisplayDf = pd.concat([fraudDf, nonFraudDf], axis=0)

    newDf = normDisplayDf.sample(frac=1)
    return newDf


def corrHeatmapVis(df, num, i, mask, cmap=''):
    plt.subplot(num, 1, i+1)
    plt.title('correlation heatmap')
    sns.heatmap(df.corr(), cmap = 'coolwarm_r', mask=mask, linewidths=0.5)
