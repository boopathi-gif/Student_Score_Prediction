import joblib
import pandas  as pd

model = joblib.load("model/student_model.pkl")
new_student_data = pd.DataFrame({"study_hours": [5], "attendance": [80], "previous_marks": [70]})
prediction = model.predict(new_student_data)

if prediction[0] == 1:
    print("The student is predicted to Pass.")
else:
    print("The student is predicted to Fail.")
    