from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lime.lime_tabular import LimeTabularExplainer
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

origins = ["http://127.0.0.1:8000"]  # replace with your frontend URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # safe now
    allow_methods=["*"],
    allow_headers=["*"]
)


model=joblib.load('model3.pkl')
X_train=joblib.load('trainexp.pkl')
print(X_train.shape)
explainer=LimeTabularExplainer(training_data=X_train.values,feature_names=X_train.columns,mode='classification',random_state=42)


class inputdata(BaseModel):
	Family_Income_Monthly:int
	Current_CGPA:float
	Physical_Verification_Score:int
	Siblings_Count:int
	Parents_Employed:int
	School_Type_Public:int
	

def correct_feature(X_train,exp):
    	feature_names = X_train.columns.tolist()
    	clean_result = []
    	for feat_cond, val in exp.as_list():
    	    for feat in feature_names:
    	        if feat in feat_cond:
    	        	clean_result.append({feat: round(val, 2)})
    	        	break
    	return clean_result

	

@app.get("/")
def home():
    return {"message": "welcome to dropout prediction"}
    
    
@app.post('/predict')
def predict(data:inputdata):
    
    income=data.Family_Income_Monthly
    cgpa=data.Current_CGPA
    pvscore=data.Physical_Verification_Score
    sibilings_count=data.Siblings_Count
    parents_employeed=data.Parents_Employed
    school=data.School_Type_Public
    
     
    df=pd.DataFrame([[income,cgpa,pvscore,sibilings_count,parents_employeed,school]],columns=['Family_Income_Monthly','Current_CGPA','Physical_Verification_Score','Siblings_Count','Parents_Employed','School_Type_Public'])
    
    
    
    prob=model.predict_proba(df)[:,1][0]
    dropout_likelihood=int(prob*100)
    retension_score=100-dropout_likelihood
    
    exp=explainer.explain_instance(data_row=df.loc[0].values,predict_fn=model.predict_proba)
    
    
    result=correct_feature(X_train,exp)
    result.extend([{"dropout_likelihood":dropout_likelihood},{"retension_score":retension_score}])    
    
    return result
    
    
    
    
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,port=8000)
    
    
    
    
    
    
    
    
    