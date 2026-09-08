import os

base = 'D:/Infotact_projects/AtmoGraph-Group-2/backend/app/services/ripple_prediction'
os.makedirs(base, exist_ok=True)

# Read the create_svc.py content which has the intended service
src = open('D:/Infotact_projects/AtmoGraph-Group-2/backend/create_svc.py').read()

# Extract the content between the triple quotes
start = src.find("content = '''") + len("content = '''")
end = src.find("'''.lstrip()")
content = src[start:end]

with open(base + '/ripple_prediction_service.py', 'w') as f:
    f.write(content.lstrip())

print('Service file created')
