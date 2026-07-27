import os
p = "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified"
print(f"Path '{p}' exists:", os.path.exists(p))
if os.path.exists(p):
    print("Contents:", os.listdir(p)[:10])
