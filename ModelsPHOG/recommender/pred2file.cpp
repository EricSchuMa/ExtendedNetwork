//
// Created by max on 17.06.19.
//

#include "phog/tree/tree.h"

#include "phog/dsl/tcond_language.h"
#include "phog/dsl/tgen_program.h"
#include "phog/model/model.h"

#include <fstream>

#include <boost/algorithm/string.hpp>

DEFINE_string(training_data, "", "A file with the tree information.");
DEFINE_string(source_file, "", "A file with the tree information");
DEFINE_string(target_file, "", "File for writing the predictions");
DEFINE_string(tgen_program_types, "", "A file with a TGen program for types.");
DEFINE_string(tgen_program_values, "",
              "A file with a TGen program for values.");
DEFINE_int32(num_training_asts, 100000,
             "Maximun number of training ASTs to load.");
DEFINE_int32(num_pred_asts, 50000,
             "Maximun number of prediction ASTs to save.");

void write_pred_to_file(TGenModel& model_types, TGenModel& model_values,
    StringSet &ss, int num_pred,
    const std::string & source_file, const std::string & target_file) {

  std::vector<TreeStorage> eval_trees;
  LOG(INFO) << "Loading evaluation data...";
  ParseTreesInFileWithParallelJSONParse(&ss, source_file.c_str(), 0,
                                        FLAGS_num_pred_asts, true, &eval_trees);
  LOG(INFO) << "Evaluation data with " << eval_trees.size() << " trees loaded.";

  // open target file for writing predictions
  std::ofstream predictions;
  predictions.open(target_file);
  Json::FastWriter json_writer;
  TreeNode node;
  for (size_t tree_id = 0; tree_id < eval_trees.size(); ++tree_id) {
    const TreeStorage &tree = eval_trees[tree_id];
    TCondLanguage::ExecutionForTree exec(&ss, &tree);
    // one object for each tree
    Json::Value json_val;
    for (unsigned node_id = 0; node_id < tree.NumAllocatedNodes(); ++node_id) {

      // same tree used for types and values
      FullTreeTraversal sample(exec.tree(), node_id);

      // initialize tree slices
      TreeSlice slice_types(exec.tree(), node_id,
                            !model_types.is_for_node_type());
      TreeSlice slice_values(exec.tree(), node_id,
                             !model_values.is_for_node_type());

      // generate outputs for value and type pair: <log_prob, label>
      std::pair<double, int> type_pred = model_types.GetBestLabelLogProb(
          model_types.start_program_id(), exec, sample, &slice_types);
      std::pair<double, int> value_pred = model_values.GetBestLabelLogProb(
          model_values.start_program_id(), exec, sample, &slice_values);

      // write type and value
      json_val[node_id]["type"] = ss.getString(type_pred.second);
      if (value_pred.second > 0) {
        json_val[node_id]["value"] = ss.getString(value_pred.second);

      } else {
        json_val[node_id]["value"] = "None";
      }

      // write probs
      json_val[node_id]["pvalue"] = value_pred.first;
      json_val[node_id]["ptype"] = type_pred.first;
    }
    // omit line break at last node
    if (tree_id == eval_trees.size() - 1){
      json_writer.omitEndingLineFeed();
    }
    predictions << json_writer.write(json_val);

  }
  predictions.close();

}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // initialize TGen programs
  StringSet ss;
  TCondLanguage lang(&ss);
  TGenProgram tgen_program_types, tgen_program_values;
  TGen::LoadTGen(&lang, &tgen_program_types, FLAGS_tgen_program_types);
  TGen::LoadTGen(&lang, &tgen_program_values, FLAGS_tgen_program_values);

  // Loading Training data
  std::vector<TreeStorage> trees;
  LOG(INFO) << "Loading training data...";
  ParseTreesInFileWithParallelJSONParse(&ss, FLAGS_training_data.c_str(), 0,
                                        FLAGS_num_training_asts, true, &trees);
  LOG(INFO) << "Training data with " << trees.size() << " trees loaded.";

  // Training the HOG for types with predefined TCOND-program
  LOG(INFO) << "Training types ...";
  TGenModel model_types(tgen_program_types, true);
  model_types.train(ss, trees, FLAGS_num_training_asts);
  model_types.GenerativeEndTraining();
  LOG(INFO) << "Training types done.";

  // Training the HOG for values with predefined TCOND-program
  LOG(INFO) << "Training values ...";
  TGenModel model_values(tgen_program_values, false);
  model_values.train(ss, trees, FLAGS_num_training_asts);
  model_values.GenerativeEndTraining();
  LOG(INFO) << "Training values done.";

  // write predictions from source file into target file
  write_pred_to_file(model_types, model_values, ss, 1, FLAGS_source_file,
      FLAGS_target_file);
}
