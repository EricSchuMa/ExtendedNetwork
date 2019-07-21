//
// Created by max on 17.06.19.
//

#include "base/stringset.h"
#include "phog/tree/tree.h"

#include "phog/dsl/tcond_language.h"
#include "phog/dsl/tgen_program.h"
#include "phog/model/model.h"

#include <fstream>

DEFINE_string(training_data, "", "A file with the tree information.");
DEFINE_string(evaluation_data, "", "A file with the training data.");
DEFINE_string(tgen_program_types, "", "A file with a TGen program for types.");
DEFINE_string(tgen_program_values, "",
              "A file with a TGen program for values.");
DEFINE_int32(num_training_asts, 100000,
             "Maximun number of training ASTs to load.");
DEFINE_int32(num_eval_asts, 50000,
             "Maximun number of evaluation ASTs to load.");
DEFINE_string(save_path, "",
              "Path to .json file for saving the predictions for each AST");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  StringSet ss;
  TCondLanguage lang(&ss);
  TGenProgram tgen_program_types, tgen_program_values;
  TGen::LoadTGen(&lang, &tgen_program_types, FLAGS_tgen_program_types);
  TGen::LoadTGen(&lang, &tgen_program_values, FLAGS_tgen_program_values);

  // TODO: write method for saving and loading trained models

  // Loading Training data
  std::vector<TreeStorage> trees, eval_trees;
  LOG(INFO) << "Loading training data...";
  ParseTreesInFileWithParallelJSONParse(&ss, FLAGS_training_data.c_str(), 0,
                                        FLAGS_num_training_asts, true, &trees);
  LOG(INFO) << "Training data with " << trees.size() << " trees loaded.";

  // Training the HOG for types with predefined TCOND-program
  LOG(INFO) << "Training types ...";
  TGenModel model_types(tgen_program_types, true);
  for (size_t tree_id = 0; tree_id < trees.size(); ++tree_id) {
    const TreeStorage &tree = trees[tree_id];
    TCondLanguage::ExecutionForTree exec(&ss, &tree);
    for (unsigned node_id = 0; node_id < tree.NumAllocatedNodes(); ++node_id) {
      model_types.GenerativeTrainOneSample(model_types.start_program_id(), exec,
                                           FullTreeTraversal(&tree, node_id));
      LOG_EVERY_N(INFO, FLAGS_num_training_asts * 100)
          << "Training... (logged every " << FLAGS_num_training_asts * 100
          << " samples).";
    }
  }
  model_types.GenerativeEndTraining();
  LOG(INFO) << "Training types done.";

  // Training the HOG for values with predefined TCOND-program
  LOG(INFO) << "Training values ...";
  TGenModel model_values(tgen_program_values, false);
  for (size_t tree_id = 0; tree_id < trees.size(); ++tree_id) {
    const TreeStorage &tree = trees[tree_id];
    TCondLanguage::ExecutionForTree exec(&ss, &tree);
    for (unsigned node_id = 0; node_id < tree.NumAllocatedNodes(); ++node_id) {
      model_values.GenerativeTrainOneSample(model_values.start_program_id(),
                                            exec,
                                            FullTreeTraversal(&tree, node_id));
      LOG_EVERY_N(INFO, FLAGS_num_training_asts * 100)
          << "Training... (logged every " << FLAGS_num_training_asts * 100
          << " samples).";
    }
  }
  model_values.GenerativeEndTraining();
  LOG(INFO) << "Training values done.";

  LOG(INFO) << "Loading evaluation data...";
  ParseTreesInFileWithParallelJSONParse(&ss, FLAGS_evaluation_data.c_str(), 0,
                                        FLAGS_num_eval_asts, true, &eval_trees);
  LOG(INFO) << "Evaluation data with " << eval_trees.size() << " trees loaded.";

  TreeNode node;
  std::ofstream predictions;
  std::cout << FLAGS_save_path;
  predictions.open(FLAGS_save_path);
  for (size_t tree_id = 0; tree_id < eval_trees.size(); ++tree_id) {
    predictions << "[";
    const TreeStorage &tree = eval_trees[tree_id];
    TCondLanguage::ExecutionForTree exec(&ss, &tree);
    for (unsigned node_id = 0; node_id < tree.NumAllocatedNodes(); ++node_id) {
      // same tree used for types and values
      FullTreeTraversal sample(exec.tree(), node_id);

      // initialize tree slices
      TreeSlice slice_types(exec.tree(), node_id,
                            !model_types.is_for_node_type());
      TreeSlice slice_values(exec.tree(), node_id,
                             !model_values.is_for_node_type());

      // generate outputs for value and type
      std::vector<std::pair<double, int const *>> output_types =
          model_types.GetLabelDistribution(model_types.start_program_id(), exec,
                                           sample, &slice_types, 1);
      std::vector<std::pair<double, int const *>> output_values =
          model_values.GetLabelDistribution(model_values.start_program_id(),
                                            exec, sample, &slice_values, 1);

      // write type-predictions to file
      predictions << "{"
                  << "\"" << "id" << "\":\"" << node_id << "\""
                  << ",\"type\":[";
      for (size_t j = 0; j < output_types.size(); ++j) {
        std::string prediction = ss.getString(*output_types[j].second);
        if (!prediction.empty()) {
          if (j < output_types.size() - 1) {
            predictions << "\"" << prediction << "\""
                        << ",";
          }
          else {
            predictions << "\"" << prediction << "\"";
          }
        }
      }

      // write value-predictions to file
      predictions << "],\"value\":[";
      for (size_t j = 0; j < output_values.size(); ++j) {
        std::string prediction = ss.getString(*output_values[j].second);
        if (!prediction.empty()) {
          if (j < output_values.size() - 1) {
            predictions << "\"" << prediction << "\""
                        << ",";
          }
          else {
            predictions << "\"" << prediction << "\"";
          }
        }
      }

      //write probabilities for type
      predictions << "],\"prob_types\":[";
      for (size_t k = 0; k < output_types.size(); ++k) {
        double prob = output_types[k].first;
        if (k < output_types.size() - 1) {
          predictions << prob << ",";
        } else {
          predictions << prob;
        }
      }
      predictions << "]}";

      //write probabilities for values
      predictions << ",\"prob_values\":[";
      for (size_t k = 0; k < output_values.size(); ++k) {
        double prob = output_values[k].first;
        if (k < output_values.size() - 1) {
          predictions << prob << ",";
        } else {
          predictions << prob;
        }
      }
      predictions << "]}";
    }
    predictions << "]";

    // write every tree in one line
    if (tree_id < eval_trees.size() - 1) {
      predictions << "\n";
    }
  }

  // TODO: develop API which accepts an AST and returns a prediction
}
