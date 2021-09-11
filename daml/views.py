from django.http import HttpResponse
from django.shortcuts import render
def index(request):
    return render(request,"index.html")
def model(request):
    return render(request,"models.html")
def analyse(request):
    longitude=float(request.POST['longitude'])
    latitude=float(request.POST["latitude"])
    housing_median_age=float(request.POST["housing_median_age" ])
    total_rooms=float(request.POST["total_rooms"])
    total_bed_rooms=float(request.POST["total_bed_rooms"])
    population=float(request.POST["population"])
    households=float(request.POST["households"])
    median_income=float(request.POST["median_income"])
    rooms_per_household=float(request.POST["rooms_per_household"])
    population_per_household=float(request.POST["population_per_household"])
    bedrooms_per_room=float(request.POST["bedrooms_per_room"])
    OCEAN =float(request.POST["OCEAN"])
    INLAND=float(request.POST["INLAND"])
    import numpy as np
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    data=pd.read_csv('city_day.csv')
    data.interpolate(limit_direction='both', inplace=True)
    data_box=data.drop(['Date','City','AQI_Bucket'],axis=1)
    data_res=data['City']
    from scipy import stats
    data_box[(np.abs(stats.zscore(data_box)) < 3).all(axis=1)]
    data=pd.concat((data_res,data_box),axis=1)
    from sklearn.model_selection import StratifiedShuffleSplit
    split  = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=29)
    for train_index,test_index in split.split(data,data["City"]):
        strat_train_set=data.loc[train_index]
        strat_test_set=data.loc[test_index]
    from sklearn import preprocessing 
    label_encoder = preprocessing.LabelEncoder() 
    strat_train_set['City']= label_encoder.fit_transform(strat_train_set['City']) 
    strat_test_set['City']= label_encoder.fit_transform(strat_test_set['City']) 
    strat_train_set_label=strat_train_set['AQI']
    strat_test_set_label=strat_test_set['AQI']
    strat_train_set_feature=strat_train_set.drop('AQI',axis=1)
    strat_test_set_feature=strat_test_set.drop('AQI',axis=1)
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler()
    strat_train_set_scaled=std.fit_transform(strat_train_set_feature)
    strat_test_set_scaled=std.fit_transform(strat_test_set_feature) 
    from sklearn.ensemble import RandomForestRegressor
    forest_reg=RandomForestRegressor()
    forest_reg.fit(strat_train_set_feature,strat_train_set_label)
    feature=([[longitude,latitude,housing_median_age,total_rooms,total_bed_rooms,population,households,median_income,rooms_per_household,population_per_household,bedrooms_per_room,OCEAN ,INLAND]])
    a=forest_reg.predict(feature)
    b={"result":a}
    print(a)
    return render(request,"analyse.html",b)

def svf(request):
    import numpy as np
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    data=pd.read_csv('city_day.csv')
    data.interpolate(limit_direction='both', inplace=True)
    data_box=data.drop(['Date','City','AQI_Bucket'],axis=1)
    data_res=data['City']
    from scipy import stats
    data_box[(np.abs(stats.zscore(data_box)) < 3).all(axis=1)]
    data=pd.concat((data_res,data_box),axis=1)
    from sklearn.model_selection import StratifiedShuffleSplit
    split  = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=29)
    for train_index,test_index in split.split(data,data["City"]):
        strat_train_set=data.loc[train_index]
        strat_test_set=data.loc[test_index]
    from sklearn import preprocessing 
    label_encoder = preprocessing.LabelEncoder() 
    strat_train_set['City']= label_encoder.fit_transform(strat_train_set['City']) 
    strat_test_set['City']= label_encoder.fit_transform(strat_test_set['City']) 
    strat_train_set_label=strat_train_set['AQI']
    strat_test_set_label=strat_test_set['AQI']
    strat_train_set_feature=strat_train_set.drop('AQI',axis=1)
    strat_test_set_feature=strat_test_set.drop('AQI',axis=1)
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler()
    strat_train_set_scaled=std.fit_transform(strat_train_set_feature)
    strat_test_set_scaled=std.fit_transform(strat_test_set_feature) 
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(strat_train_set_feature,strat_train_set_label)
    d=regressor.score(strat_train_set_feature,strat_train_set_label)
    return render(request,'models.html',{"d":d})

def decisionTree(request):
    import numpy as np
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    data=pd.read_csv('city_day.csv')
    data.interpolate(limit_direction='both', inplace=True)
    data_box=data.drop(['Date','City','AQI_Bucket'],axis=1)
    data_res=data['City']
    from scipy import stats
    data_box[(np.abs(stats.zscore(data_box)) < 3).all(axis=1)]
    data=pd.concat((data_res,data_box),axis=1)
    from sklearn.model_selection import StratifiedShuffleSplit
    split  = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=29)
    for train_index,test_index in split.split(data,data["City"]):
        strat_train_set=data.loc[train_index]
        strat_test_set=data.loc[test_index]
    from sklearn import preprocessing 
    label_encoder = preprocessing.LabelEncoder() 
    strat_train_set['City']= label_encoder.fit_transform(strat_train_set['City']) 
    strat_test_set['City']= label_encoder.fit_transform(strat_test_set['City']) 
    strat_train_set_label=strat_train_set['AQI']
    strat_test_set_label=strat_test_set['AQI']
    strat_train_set_feature=strat_train_set.drop('AQI',axis=1)
    strat_test_set_feature=strat_test_set.drop('AQI',axis=1)
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler()
    strat_train_set_scaled=std.fit_transform(strat_train_set_feature)
    strat_test_set_scaled=std.fit_transform(strat_test_set_feature) 
    from sklearn.tree import DecisionTreeRegressor  
    Tree_regressor = DecisionTreeRegressor(random_state = 0)  
    Tree_regressor.fit(strat_train_set_feature,strat_train_set_label) 
    b=Tree_regressor.score(strat_train_set_feature,strat_train_set_label)
    return render(request,'models.html',{"b":b})

def linear(request):
    import numpy as np
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    data=pd.read_csv('city_day.csv')
    data.interpolate(limit_direction='both', inplace=True)
    data_box=data.drop(['Date','City','AQI_Bucket'],axis=1)
    data_res=data['City']
    from scipy import stats
    data_box[(np.abs(stats.zscore(data_box)) < 3).all(axis=1)]
    data=pd.concat((data_res,data_box),axis=1)
    from sklearn.model_selection import StratifiedShuffleSplit
    split  = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=29)
    for train_index,test_index in split.split(data,data["City"]):
        strat_train_set=data.loc[train_index]
        strat_test_set=data.loc[test_index]
    from sklearn import preprocessing 
    label_encoder = preprocessing.LabelEncoder() 
    strat_train_set['City']= label_encoder.fit_transform(strat_train_set['City']) 
    strat_test_set['City']= label_encoder.fit_transform(strat_test_set['City']) 
    strat_train_set_label=strat_train_set['AQI']
    strat_test_set_label=strat_test_set['AQI']
    strat_train_set_feature=strat_train_set.drop('AQI',axis=1)
    strat_test_set_feature=strat_test_set.drop('AQI',axis=1)
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler()
    strat_train_set_scaled=std.fit_transform(strat_train_set_feature)
    strat_test_set_scaled=std.fit_transform(strat_test_set_feature) 
    from sklearn.linear_model import LinearRegression
    lin_reg=LinearRegression()
    lin_reg.fit(strat_train_set_feature,strat_train_set_label)
    a=lin_reg.score(strat_train_set_feature,strat_train_set_label)
    return render(request,'models.html',{"a":a})

    
def RandomForest(request):
    import numpy as np
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    data=pd.read_csv('city_day.csv')
    data.interpolate(limit_direction='both', inplace=True)
    data_box=data.drop(['Date','City','AQI_Bucket'],axis=1)
    data_res=data['City']
    from scipy import stats
    data_box[(np.abs(stats.zscore(data_box)) < 3).all(axis=1)]
    data=pd.concat((data_res,data_box),axis=1)
    from sklearn.model_selection import StratifiedShuffleSplit
    split  = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=29)
    for train_index,test_index in split.split(data,data["City"]):
        strat_train_set=data.loc[train_index]
        strat_test_set=data.loc[test_index]
    from sklearn import preprocessing 
    label_encoder = preprocessing.LabelEncoder() 
    strat_train_set['City']= label_encoder.fit_transform(strat_train_set['City']) 
    strat_test_set['City']= label_encoder.fit_transform(strat_test_set['City']) 
    strat_train_set_label=strat_train_set['AQI']
    strat_test_set_label=strat_test_set['AQI']
    strat_train_set_feature=strat_train_set.drop('AQI',axis=1)
    strat_test_set_feature=strat_test_set.drop('AQI',axis=1)
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler()
    strat_train_set_scaled=std.fit_transform(strat_train_set_feature)
    strat_test_set_scaled=std.fit_transform(strat_test_set_feature) 
    from sklearn.ensemble import RandomForestRegressor
    forest_reg=RandomForestRegressor()
    forest_reg.fit(strat_train_set_feature,strat_train_set_label)
    c=forest_reg.score(strat_train_set_feature,strat_train_set_label)
    return render(request,'models.html',{"c":c})
