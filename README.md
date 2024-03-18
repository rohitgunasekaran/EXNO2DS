# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic.csv")
dt
```
![312083120-e06452cb-fff5-46d9-91e5-1e13daed721d](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/4e051c84-3ef7-4798-bee6-aca9d4618ab0)
```
dt.info()
```
![312083583-8bde848f-e44f-44a9-ac0f-e259f11c65a7](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/30868b0e-b795-4fc9-9ed8-9ac7a31e4ec7)
```
dt.set_index('PassengerId',inplace=True)
dt.describe()
```
![312084768-813b0217-8139-4f0f-bb4c-7cef93046ca5](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/863b1af9-4fe8-4b3a-9c5b-031231dc4b68)
```
dt.shape
```
![312084953-fe9f5eef-962b-4984-9f62-0928129a36d6](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/c54585b6-defa-4896-a665-c0aa221303b9)
```
dt.nunique()
```

![312085145-428f6980-9a18-48f0-9c0a-cc1f8090bee1](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/2e0c3c6c-37fe-4971-aabf-a369ac945730)
```
dt["Survived"].value_counts()
```

![312085389-6b13cbf2-e40f-477b-854c-6b035f60eb08](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/cfa5f803-1660-41ae-8f53-d6f816849317)
```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![312085594-c6951eef-9034-4f9f-9f1a-8724a018dbea](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/03cdd245-9048-4127-8ca9-c32ef8346eea)
```
sns.countplot(data=dt,x="Survived")
```
![312085841-07df1378-7f18-4d9d-b891-2f839ccb05b0](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/a1805f35-d617-4db7-8ac4-68de8ad589e4)
```
dt.Pclass.unique()
```
![312086072-c421cbe0-9718-48cf-812a-266a17c27ff9](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/83772692-11f1-4e86-9e6d-4d580daaa813)
```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![312086422-76377e69-143a-4fdf-937a-d3d0260ad75b](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/e81ad4bb-99b5-4970-8846-c4d5086f8310)
```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![312086721-5269a77c-0399-4250-aec3-228d0ab6d31c](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/a079cbbc-4be4-4bce-a753-496226cc776e)
```
sns.catplot(x='Survived',hue="Gender",data=dt,kind="count")
```
![312086960-cf409002-66a2-49c1-bab9-daed33d5d5f9](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/0cf2efd3-cf46-4a7d-b2ca-2b1baa2fdfba)

```
dt.boxplot(column="Age",by="Survived")
```

![312087277-7de0dd77-83ba-47c1-aac4-292b6f552717](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/bdc5194b-40e1-402a-9f57-2c6281a5850e)
```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![312087598-f44a4475-948b-4070-87e8-5b5c1cf03ea6](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/3c454a46-7da6-4a00-aaad-84a0d722d31e)

```
sns.jointplot(x=dt["Age"],y=dt["Fare"],data=dt)
```
![312088001-73f0aaf9-cd78-411c-b8af-5e857c0db614](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/c126e649-1f51-4a1b-9fe7-560c83881b5c)
```
fig,ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=dt)
```
![312088316-8b2d554b-a089-4e15-8b76-29f26fe6f40f](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/2ad84b3f-3e39-4292-942e-5c9d5ce75b0e)
```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![312088646-570d378c-9f74-4ec8-9a20-df375e105912](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/f13e9a90-7141-4b34-8113-ffeb80b41253)
```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![312088959-4b2293e7-e5d7-41fe-b0b1-64c59e5cfd49](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/d6145544-4b82-45af-99da-8847f15abd76)
```
sns.pairplot(dt)
```
![312089950-4de84806-e138-419d-b9d9-776f376baff5](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/7014040a-ec3c-4002-84f2-1b9711121b2a)
![312090357-b3426fc2-b11f-4178-a30e-064d6f3a37a3](https://github.com/aparnabalasubrmanian/EXNO2DS/assets/123351172/355fb56d-74c3-49c0-bf58-147bd4329140)

# RESULT:
         thus data analyzing of the given data is successful
       
