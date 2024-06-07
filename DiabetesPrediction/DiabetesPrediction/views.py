from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request,'predict.html')
def result(request):
    data = pd.read_csv(r'C:\Users\mitda\Downloads\archive\diabetes.csv')

    X = data.drop("Outcome",axis = 1)
    Y = data["Outcome"]
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y,test_size = 0.1,stratify=Y,random_state=66)

    RandomClassifier = RandomForestClassifier(n_estimators=100,criterion='gini',oob_score=True)
    RandomClassifier.fit(X_train,Y_train)
    
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    predictions = RandomClassifier.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
    result1 = ""
    if predictions == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"
    return render(request,'predict.html',{"result2":result1,"n1":val1,"n2":val2,"n3":val3,"n4":val4,"n5":val5,"n6":val6,"n7":val7,"n8":val8})