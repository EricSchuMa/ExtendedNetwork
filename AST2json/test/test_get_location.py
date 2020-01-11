import json as json

file = "../json/python50k_eval.json"

class TestLocation():

    def test_get_location(self):
        with open(file) as f:
            for line in f:
                data = json.loads(line)
                for idx, dict in enumerate(data):
                    file_id, lineno, node_id = dict["location"]
                    print("File id is: {}".format(file_id))
                    print("Line number is: {}".format(lineno))
                    print("Node id is: {}".format(node_id))