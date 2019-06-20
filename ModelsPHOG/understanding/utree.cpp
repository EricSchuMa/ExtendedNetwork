//
// Created by max on 17.06.19.
//

#include "phog/tree/tree.h"
#include "base/stringset.h"

#include "phog/dsl/tcond_language.h"
#include "phog/dsl/tgen_program.h"
#include "phog/model/model.h"

DEFINE_string(training_data, "", "A file with the tree information.");
DEFINE_string(evaluation_data, "", "A file with the training data.");
DEFINE_string(tgen_program, "", "A file with a TGen program.");
DEFINE_bool(is_for_node_type, true, "Whether the predictions are for node type (if false it is for node value).");
DEFINE_int32(num_training_asts, 100000, "Maximun number of training ASTs to load.");
DEFINE_int32(num_eval_asts, 50000, "Maximun number of evaluation ASTs to load.");


int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    StringSet ss;
    TCondLanguage lang(&ss);
    TGenProgram tgen_program;
    TGen::LoadTGen(&lang, &tgen_program, FLAGS_tgen_program);

    std::vector<TreeStorage> trees, eval_trees;
    LOG(INFO) << "Loading training data...";
    ParseTreesInFileWithParallelJSONParse(
            &ss, FLAGS_training_data.c_str(), 0, FLAGS_num_training_asts, true, &trees);
    LOG(INFO) << "Training data with " << trees.size() << " trees loaded.";

    LOG(INFO) << "Loading evaluation data...";
    ParseTreesInFileWithParallelJSONParse(
            &ss, FLAGS_evaluation_data.c_str(), 0, FLAGS_num_eval_asts, true, &eval_trees);
    LOG(INFO) << "Evaluation data with " << eval_trees.size() << " trees loaded.";

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


    TreeNode node;
    for (size_t tree_id = 0; tree_id < eval_trees.size(); ++tree_id) {
        const TreeStorage& tree = eval_trees[tree_id];
        TCondLanguage::ExecutionForTree exec(&ss, &tree);
        for (unsigned node_id = 0; node_id < tree.NumAllocatedNodes(); ++node_id) {
            FullTreeTraversal sample(exec.tree(), node_id);
            TreeSlice slice(exec.tree(), node_id, !model.is_for_node_type());
            std::vector<std::pair<double, int const*>> output = model.GetLabelDistribution(model.start_program_id(), exec, sample, &slice);

            //for (size_t i = 0; i<output.size(); ++i) {
               // std::cout <<output[i].first <<  "           "<<*output[i].second << std::endl;
            //}
            std::cout << "Probability: " << output[0].first << " Prediction: " << ss.getString(*output[0].second) << " Truth: " << ss.getString(tree.node(node_id).Type()) << std::endl;

            //std::cout << tree.node(node_id).Type() << std::endl;

            //std::cout << tree.node(node_id).Type() << " --- " <<  model.GetLabelAtPosition(model.start_program_id(),
              //                                                                 exec, sample, &slice, false) << std::endl;

        }
    }

}
