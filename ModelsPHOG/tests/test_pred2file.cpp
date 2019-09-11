//
// Created by max on 29.07.19.
//

#include "base/readerutil.h"
#include "json/json.h"
#include <iostream>
#include <string>
#include <memory>
#include <assert.h>

DEFINE_string(filename_pred,
    "/home/max/Documents/bachelorThesis/"
    "ExtendedNetwork/data/phog_pred_50k_eval.json",
    "file with the predictions");
DEFINE_string(filename_original,
    "/home/max/Documents/bachelorThesis/"
    "ExtendedNetwork/data/python50k_eval.json",
    "file with original ASTs");

int main(int argc, char** argv){
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::unique_ptr<RecordInput> input_pred(new FileRecordInput(
      FLAGS_filename_pred));
  std::unique_ptr<InputRecordReader> reader_pred(input_pred->CreateReader());

  std::unique_ptr<RecordInput> input_orig(new FileRecordInput(
      FLAGS_filename_original));
  std::unique_ptr<InputRecordReader> reader_orig(input_orig->CreateReader());

  const std::string filename_errors = \
  "/home/max/Documents/bachelorThesis/ExtendedNetwork/data/errors.txt";
  std::ofstream errors;
  errors.open(filename_errors);

  Json::Reader jsonreader;
  std::map<std::string, double> accuracies;
  accuracies["type"] = 0;
  accuracies["value"] = 0;
  int num_asts = 0;
  while (!reader_pred->ReachedEnd() && !reader_orig->ReachedEnd()) {
    ++ num_asts;
    std::string s_pred;
    reader_pred->Read(&s_pred);

    std::string s_orig;
    reader_orig->Read(&s_orig);

    Json::Value v_pred;
    Json::Value v_orig;
    jsonreader.parse(s_pred, v_pred, false);
    jsonreader.parse(s_orig, v_orig, false);

    // check if all nodes have been registered
    // and predictions for each node exist
    assert(v_orig.size() == v_pred.size());

    // test prediction accuracy
    double total = v_orig.size();

    for (auto& key: {"type", "value"}) {
      double correct_pred = 0;
      for (unsigned int j = 0; j < v_orig.size(); ++j) {
        if (v_orig[j][key] == v_pred[j][key]) {
          correct_pred++;
        }
        else if (v_orig[j][key].empty() && v_pred[j][key] == "None") {
          correct_pred++;
        }
        else{
          errors << "Value/Type: " << key
          <<"Prediction: " << v_pred[j][key]
          << "Ground Truth: " << v_orig[j][key] << std::endl;
        }

      }
      accuracies[key] += (correct_pred / total);
    }
  }
  accuracies["type"] /= num_asts;
  accuracies["value"] /= num_asts;

  std::cout << "\nAccuracy value = " << accuracies["value"]
  << "\nAccuracy type = " << accuracies["type"];

  // accuracies should be over 80% for types and over 60% for values
  assert(accuracies["type"] > 0.8);
  assert(accuracies["value"] > 0.6);
  std::cout << "\nAll tests passed";
}