//
// Created by max on 17.06.19.
//

#include "phog/tree/tree.h"
#include "base/stringset.h"

#include "phog/dsl/tcond_language.h"
#include "phog/dsl/tgen_program.h"
#include "phog/model/model.h"

#include <fstream>

DEFINE_string(training_data, "", "A file with the tree information.");
DEFINE_string(evaluation_data, "", "A file with the training data.");
DEFINE_string(tgen_program, "", "A file with a TGen program.");
DEFINE_bool(is_for_node_type, false, "Whether the predictions are for node type (if false it is for node value).");
DEFINE_int32(num_training_asts, 100000, "Maximun number of training ASTs to load.");
DEFINE_int32(num_eval_asts, 50000, "Maximun number of evaluation ASTs to load.");
DEFINE_string(save_path, "", "Path to .json file for saving the predictions for each AST");


int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    StringSet ss;
    TCondLanguage lang(&ss);
    TGenProgram tgen_program;
    TGen::LoadTGen(&lang, &tgen_program, FLAGS_tgen_program);

    //TODO: write class for saving and loading trained models

    // Loading Training data
    std::vector<TreeStorage> trees, eval_trees;
    LOG(INFO) << "Loading training data...";
    ParseTreesInFileWithParallelJSONParse(
            &ss, FLAGS_training_data.c_str(), 0, FLAGS_num_training_asts, true, &trees);
    LOG(INFO) << "Training data with " << trees.size() << " trees loaded.";

    // Training the HOG with predefined TCOND-program
    LOG(INFO) << "Training...";
    TGenModel model(tgen_program, FLAGS_is_for_node_type);
    for (size_t tree_id = 0; tree_id < trees.size(); ++tree_id) {
        const TreeStorage &tree = trees[tree_id];
        TCondLanguage::ExecutionForTree exec(&ss, &tree);
        for (unsigned node_id = 0; node_id < tree.NumAllocatedNodes(); ++node_id) {
            model.GenerativeTrainOneSample(model.start_program_id(), exec, FullTreeTraversal(&tree, node_id));
            LOG_EVERY_N(INFO, FLAGS_num_training_asts * 100)
                << "Training... (logged every " << FLAGS_num_training_asts * 100 << " samples).";
        }
    }
    model.GenerativeEndTraining();
    LOG(INFO) << "Training done.";


    LOG(INFO) << "Loading evaluation data...";
    ParseTreesInFileWithParallelJSONParse(
            &ss, FLAGS_evaluation_data.c_str(), 0, FLAGS_num_eval_asts, true, &eval_trees);
    LOG(INFO) << "Evaluation data with " << eval_trees.size() << " trees loaded.";

    TreeNode node;
    std::ofstream predictions;
    std::cout << FLAGS_save_path;
    predictions.open(FLAGS_save_path);
    for (size_t tree_id = 0; tree_id < eval_trees.size(); ++tree_id) {
        predictions << "[";
        const TreeStorage& tree = eval_trees[tree_id];
        TCondLanguage::ExecutionForTree exec(&ss, &tree);
        for (unsigned node_id = 0; node_id < tree.NumAllocatedNodes(); ++node_id) {
            predictions << "{" << "\"id\":\""<< node_id << "\"" << ",\"predictions\":[";
            FullTreeTraversal sample(exec.tree(), node_id);
            TreeSlice slice(exec.tree(), node_id, !model.is_for_node_type());
            std::vector<std::pair<double, int const*>> output = model.GetLabelDistribution(model.start_program_id(),
                                                                                           exec, sample, &slice);
            for (size_t j = 0; j<output.size(); ++j) {
                std::string prediction = ss.getString(*output[j].second);
                if (prediction != "") {
                    if (j < output.size() - 1) {
                        predictions << "\"" << prediction << "\"" << ",";
                    } else {
                        predictions << "\"" << prediction << "\"";
                    }
                }
            }
            predictions << "],\"probabilities\":[";
            for (size_t k = 0; k<output.size(); ++k) {

                double prob = output[k].first;
                if (k<output.size()-1){
                    predictions << prob << ",";
                }
                else{
                    predictions << prob;
                }
            }
            predictions << "]}";
        }
        predictions << "]";
        predictions.close();
    }

    // TODO: develop API which accepts an AST and returns a prediction


}
