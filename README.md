# PC_SF_ML
Submission for ML tasks
I have dropped some features due to too many missing values or because I combined multiple features into one. Below is the list of all the modifications I performed on given features
input:
dropped features: (due to missing values) ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Id']
filling null values: (filled with mean) ['LotFrontage', 'MasVnrArea', 'GarageYrBlt'] (filled with mode) ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
Dropped features (whose value may be approximated using other features):['Condition1', 'Condition2','HouseStyle','RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea','BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinSF1','BsmtFinSF2', 'Heating','MoSold', 'SaleType','SaleCondition','Electrical','Street','GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars','Foundation','Functional']
LabelEncoding for features: category_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, 'NA':0,'Y':2,'N':0,'P':1,'Gtl':1,'Mod':2,'Sev':3}
df_red['KitchenQual_encoded'] = encoder.fit_transform(df_red['KitchenQual'])
  df_red['ExterQual_encoded'] = encoder.fit_transform(df_red['ExterQual'])
  df_red['Extercond_encoded'] = encoder.fit_transform(df_red['ExterCond'])
  df_red['GarageQual_encoded'] = encoder.fit_transform(df_red['GarageQual'])
  df_red['Garagecond_encoded'] = encoder.fit_transform(df_red['GarageCond'])
  df_red['PavedDrive_encoded'] = encoder.fit_transform(df_red['PavedDrive'])
  df_red['LandSlope'] = encoder.fit_transform(df_red['LandSlope'])
  df_red['HeatingQC'] = encoder.fit_transform(df_red['HeatingQC'])
  df_red['CentralAir'] = encoder.fit_transform(df_red['CentralAir'])
Combining features: df_red['house_cond'] = df_red['1stFlrSF']+df_red['2ndFlrSF']-(0.5)*df_red['LowQualFinSF']+df_red['GrLivArea'] + (50)*df_red['BsmtFullBath']+ (50)*df_red['BsmtHalfBath']+ (50)*df_red['FullBath']+ (50)*df_red['HalfBath']
  + (500)*df_red['KitchenAbvGr']*df_red['KitchenQual_encoded'] + 100*df_red['TotRmsAbvGrd']
  + 100*df_red['Fireplaces'] + df_red['WoodDeckSF']+ df_red['OpenPorchSF']+ df_red['EnclosedPorch']+ df_red['3SsnPorch'] +df_red['ScreenPorch']+ df_red['PoolArea'] + df_red['MiscVal'] + df_red['LotArea'] + 10*df_red['LotFrontage'] + 100*(df_red['OverallQual']+df_red['OverallCond']+df_red['ExterQual_encoded']+df_red['Extercond_encoded']+df_red['CentralAir']+df_red['HeatingQC']) + df_red['TotalBsmtSF'] + df_red['GarageArea']
  df_red['Garage_cond'] = df_red['GarageQual_encoded'] + df_red['Garagecond_encoded'] + df_red['PavedDrive_encoded'] - df_red['LandSlope']
  df_red['Age'] = df_red['YrSold'] - (df_red['YearRemodAdd'] + df_red['YearBuilt'])*0.5
dropping duplicate features: df_red.drop(['LotFrontage', 'LotArea','1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd','Fireplaces','WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
        'MiscVal', 'KitchenQual_encoded','OverallQual','OverallCond','ExterQual_encoded',
  'Extercond_encoded','BsmtFinType1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF','ExterQual', 'ExterCond', 'GarageQual', 
  'GarageCond', 'GarageQual_encoded', 'Garagecond_encoded','GarageArea','PavedDrive','PavedDrive_encoded','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'BldgType','CentralAir', 'HeatingQC', 'YearBuilt', 'YearRemodAdd','YrSold'], axis=1,inplace=True)
Encoded some features with mean values: df_red["NHB_encoded"] = df_red.groupby("Neighborhood")["SalePrice"].transform("mean")
  df_red["MSZ_encoded"] = df_red.groupby("MSZoning")["SalePrice"].transform("mean")
Scaled all these features with StandardScaler to fit in a KNN Regressor
Finally there are 6 input features left :MSSubClass,	house_cond,	Garage_cond,	Age,	NHB_encoded,	MSZ_encoded.
