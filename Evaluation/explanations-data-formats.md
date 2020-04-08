* Explanations of the result formats and encodings used in the project.
* Artur Andrzejak, Tuyen Le (January/March 2020)

# Evaluation

## Overview of the evaluation
Analysis of results consists of the following steps:
1. During preprocessing (specifically, running get_terminal_extended.py) we create a file "node_extra_info" with 
extended ground truth data.  
2. During computing of predictions (running evaluation_with_loc.py) we create a log file with results of predictions (results_log file).
3. Both files are merged in the 3rd step (running result_log_analysis_refactored.py), and cached. Further analysis with the cached file computes entries in the Table 4 (later Table 4 and 5) in the paper.

###Note on terminal dictionary
Terminal dictionary (terminal_dict() is loaded from terminal_dict_filename.
*    We use "1k" or "10k" size (effectively 1000 or 10000).
*    Index 0 belongs to "EmptY" which means that the node has a non-terminal value.
*    => Effectively we have indices 1..999 (for 1k), or 1..99999 (for 10k).


## Detailed information on evaluation files and code

###Information on step 1: node_extra_info
The file "node_extra_info" with extended ground truth data is created when running running get_terminal_extended.py.
Explanations of columns in node_extra_info:
+ 'file_id', 'src_line', 'ast_node_idx': location of the AST node
+ 'has_terminal': is true if node has field "value", i.e. node has terminal value
+ 'in_dict': is true if value of node is in terminal_dict (the truth value is in terminal_dict)
+ 'in_attn_window': is true if value of node is in attn_queue (the truth value is in attn_queue)
+ 'phog_ok': is true if value of node in test data is the same with value of node in corresponding hog data. In other words, PHOG has predicted correctly
+ 'ast_idx' = 'file_id'
+ 'node_idx' = 'ast_node_idx'

####How we get this while running get_terminal_extended.py:
Infos from get_terminal_extended.process():
+ [line 67] node_truths.has_terminal = 'value' in dic
+ [line 68] node_truths.ast_idx = ast_index
+ [line 69] node_truths.node_idx = node_idx
+ [line 73] node_truths.in_dict = dic_value in terminal_dict
+ [line 74] node_truths.in_attn_window = dic_value in attn_que
+ [line 75] node_truths.phog_ok = (dic_value == dic_hog["value"])

###Information on step 2: results_log
Here we explain the meaning of columns in the results_log file (created during predictions), and meaning of their values. 

####Explanations of columns in results_log table
These are the prediction results with location info, currently 'results_log_Mar_5_1820.csv'.

* "prediction_idx": index of current predictions (for 50k eval data and 10k dict, 0..31,692,799) <!--(0.. 6.5 millions)-->
* "epoch_num": prediction epoch, for 50k eval data and 10k dict, 0..4,951 <!--0...1015 (each has 6400 predictions)-->
* "**truth**": encoding of the ground truth (see extended description below)
* "_prediction_": encoding of the prediction (see extended description below) 
* "_new_prediction_": summary of the prediction result by Max, with:
    * 0: pred == hogID
    * 1: pred == unkID
    * 2: pred == label (= truth)
    * 3: otherwise
* "file_id", "src_line", "ast_node_idx": location of the AST node


#### Meaning of _ground truth_ values in results_log file
The constants used to for the value of the column "truth" (in code it is "label") have the following meaning (for their concrete value see information on step 3 below): 
* EmptY_idx:  AST node has no terminal value (and we predict only terminal values)
* tdict_start_idx..tdict_end_idx: the terminal value is in the terminal dictionary, so RNN should be able to predict.
* unk_id: this means that pointer network CANNOT predict (not in dict, 
		not in attention window), AND phog predicted wrongly.
* hog_id: this means that PHOG predicted correctly BUT RNN and pointer net failed.
* eof_idx: constant for the padding in preprocessing, just ignore.
* attn_start_idx...attn_end_idx: token can be found in the attention windown, with _location_index_ := number of tokens to go back to get a previous copy of current token = (*value*-(attn_start_idx)+1). 
    * Example: dic_value = 'TemplateSpec', i = 7 (node_idx), att_que {7} = [.., 'TemplateSpec', 'Simple', 'EmptY']. 
        * We have location_index = 3 and <value> = 1005 (in code *value* = location_id).  

##### Code locations which give us the interpretation of the ground truth values
Infos from preprocess_code.get_terminal_extended.process():
* [line: 30]  unk_id: set this const to the first ID "above" the ID-range of the terminal_dict
* [line: 48]  hog = unk_id + 1
* [line: 74]  order of constant values is [<seq_dict>] [unk, hog_id, eof] [<loc_idx>]
* [line: 74]  location_id = unk_id + 2 + location_index  # [unk, hog_id, eof, loc_idx]
* [line: 141] attn_size = 50
* [line: 148] process( ..., unk_id=vocab_size, ...)
* [line: 48]  location_index = 
        [len(attn_que)-ind for ind, x in enumerate(attn_que) if x == dic_value][-1]


#### Meaning of the _prediction values_ in results_log file
The results_log file has two prediction-related values:
* "_prediction_": encoding of the prediction as returned by NN/TensorFlow 
* "_new_prediction_": a summary of the prediction result computed by Max (explanation below).

To understand the prediction value ("_prediction_") we need to understand how the TF model predicts (this is a probably guess since we cannot verify TF without large effort):
* The "meta-switch" of the extended-network has choice between tree prediction methods:
A. RNN, B. pointer net, C. PHOG.
##### For A (RNN)
* RNN can decide what the next token does not have value and sets prediction to "EmptY_idx" (== 0). This happens in a large number of predictions!
* RNN predics a value from the terminal dictionary [tdict_start_idx...tdict_end_idx]
* RNN decides that it cannot predict and returns unkID (== terminal_dict_size)
##### For B (Pointer net)
* If pointer net is used, the prediction will be in the range [attn_start_idx..attn_end_idx] (which has size attn_window_size: 50). This range is above the range of the terminal dict (attn_start_idx = terminal_dict_size + 3). 
##### For C (PHOG)
* If PHOG is used, the prediction output is hog_id (= terminal_dict_size + 1). **This does not tell yet whether the prediction is correct** (or does? Check this!).

#### Converted ground truth and prediction values
Max code uses the original ground truth ("label") and prediction values and transforms them as follows n evaluation_with_loc.convert_labels_and_predictions().
The purpose is to compute the confusion matrix. The transformations described below are as in code, but their interpretation is OURs (=> Check with Max).
* "_new_prediction_": derived from _prediction_:
    * 0: pred == hogID. => Prediction indicates that the PHOG-computed value should be used as a prediction. It does not say that the result is OK.
    * 1: pred == unkID. => Ext. network returned that it cannot predict (does this happen?)
    * 2: pred == label. => Prediction was done by the RNN or pointer network, and it is correct (ground truth and prediction are the same).
    * 3: otherwise. => Most likely: the prediction was done by the RNN or pointer network, and failed.

* "_new_label_": derived from _label_ (see code: `new_labels = [0 if label == hogID else 1 if label == unkID else 2` )
    * 0: pred == hogID. => The PHOG prediction is correct and should be used. 
    * 1: pred == unkID. => RNN/pointer net cannot predict, and PHOG failed to predict correctly (true?)
    * 2: pred == label. => Prediction can be done by the RNN or pointer network, the value of "_label_" indicates the actual ground truth.

##### Note on the confusion matrix used by Max
Correct PHOG prediction is counted by Max only if _new_prediction_ == _new_label_ == hogID.
However, we saw in data (row 37) that the original label was in attn-window 
(so _new_label_ == 2), and the original prediction was == hog_id. (so _new_prediction_ == hogID).
Consequently, in the confusion matrix this is counted as error.
At the same time, from node_extra_info we know that the phog prediction was ok (phog_ok == 1), 
so this prediction could have been used.
=> In our analysis we can set phog_ok = True only if prediction = truth = hogID, to be consistent
with his results.

### Information on step 3: code result_log_analysis_refactored.py
We describe here the concrete values and meaning of the ground truth values as assumed in the code "result_log_analysis_refactored.py".
Depending on the size of the terminal directory (terminal_dict_size) we get different values of the ground truth constants.
##### Note: the interpretations below do not say whether they apply to the ground truth or prediction - check!

The **terminal_dict_size** can have size 1k, 10k or 50k. We get:
* EmptY_idx = 0 (marker of nodes without terminal values)
* tdict_start_idx = 1 (1st index of the values in the terminals dict)
* tdict_end_idx = terminal_dict_size - 1 (last index of the values in the terminals dict)
* unk_id = terminal_dict_size 
* hog_id = terminal_dict_size + 1 (marker that the PHOG prediction is used or PHOG-usage predicted)
* eof_idx = terminal_dict_size + 2 (marker for padding)
* Related to pointer network results:
    + attn_window_size: 50 (parameter, size of the look-back window)
    + attn_start_idx = terminal_dict_size + 3  (first index for ("shifted") values predicted by pointer net)
    + attn_end_idx = attn_start_idx + attn_window_size (first index for ("shifted") values predicted by pointer net)


 
#Prediction_viewer
#####Meaning of inserts in source files representing prediction results for the next token.

New encoding (**dph**) | Old encoding | Meaning
---| --- | ---
`+..` | S | the prediction is true and the truth value is in [tdict_start_idx..tdict_end_idx]
`-+.` | A | the prediction is true and the truth value >= attn_start_idx
`--+` | H-truth value | the prediction is HOG
`*--` | F-prediction value-truth value | the prediction is false and the truth value is in [tdict_start_idx..tdict_end_idx]
`-*-` | G-prediction value-truth value | the prediction is false and the truth value >= attn_start_idx
`---` | U-prediction value-truth value | the prediction is false and the truth value is unk_id
`???` | HU | the prediction is HOG and the truth value is unk_id
not occur | UNK | unknown id. However, there is no UNK in the prediction values
not occur | AU | the prediction is true and the truth value is unk_id


#### Further notes on prediction viewer
Tuyen, 20-01-2020, [email](https://mail.google.com/mail/u/0/?tab=wm#inbox/KtbxLvHcLqJnHqznMZSDHsTVrxWsqtbQBB)

I have modified the prediction_viewer.py as below. please note that,
if the prediction / truth value is in [0..999], the value will be the
corresponding terminal in the terminal_dict. Otherwise, the value will be
from 1000 upwards.

I have noticed that hog_id = unk_id + 1 (i.e. hog_id is 1001 and unk_id is 1000 in this case).

 (File "2020-01-20-16-06-prediction_viewer_results.zip"):