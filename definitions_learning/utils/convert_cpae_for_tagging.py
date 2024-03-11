import json

# Opening JSON file
f = open('./data/definitions/cpae/dict_wn.json')
out_f = open("cpae_definitions.csv", "w")
  
# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list
index = 0
for definendum in data:
    for definition in data[definendum]:
        print(str(index)+";"+"none"+";"+definendum+";"+" ".join(definition).replace(";",""), file = out_f)
        index += 1