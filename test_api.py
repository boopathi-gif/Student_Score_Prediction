import requests

url = "http://127.0.0.1:8000/predict"

student_data = {"study_hours": 5, "attendance": 90, "previous_marks": 80}

response = requests.post(url, json=student_data)

print("STATUS CODE:",response.status_code)
print("PREDICTED RESULT:",response.json())

      