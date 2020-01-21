# Overview
Explanations of the result formats and encodings used in the project.

Artur Andrzejak, Tuyen Le (Jan 2020)


#For prediction_viewer
#####Meaning of inserts in source files representing prediction results for the next token.

New encoding (**dph**) | Old encoding | Meaning
---| --- | ---
`+..` | S | the prediction is true and the truth value is in [1..999]
`-+.` | A | the prediction is true and the truth value >= 1003
`--+` | H-truth value | the prediction is HOG
`*--` | F-prediction value-truth value | the prediction is false and the truth value is in [1..999]
`-*-` | G-prediction value-truth value | the prediction is false and the truth value >= 1003
`---` | U-prediction value-truth value | the prediction is false and the truth value is unk_id
`???` | HU | the prediction is HOG and the truth value is unk_id
not occur | UNK | unkown id. However, there is no UNK in the prediction values
not occur | AU | the prediction is true and the truth value is unk_id

<!---
+ "H-truth value": the prediction is HOG.
+ "S": the prediction is true and the truth value is in [0..999]
+ "AU":  the prediction is true and the truth value is unk_id 
+ "A": the prediction is true and the truth value >= 1001

+ "F-prediction value-truth value": the prediction is false and the truth value is in [0..999]
+ "G-prediction value-truth value": the prediction is false and the truth value >= 1001
+ "U-prediction value-truth value": the prediction is false and the truth value is unk_id

+ "HU": the prediction is HOG and the truth value is unk_id
+ "UNK": unkown id. However, there is no UNK in the prediction values. I checked the CSV file again.--->

#### Further notes
Tuen, 20-01-2020, [email](https://mail.google.com/mail/u/0/?tab=wm#inbox/KtbxLvHcLqJnHqznMZSDHsTVrxWsqtbQBB)

I have modified the prediction_viewer.py as below. please note that,
if the prediction / truth value is in [0..999], the value will be the
corresponding terminal in the terminal_dict. Otherwise, the value will be
from 1000 upwards.

I have noticed that hog_id = unk_id + 1 (i.e. hog_id is 1001 and unk_id is 1000 in this case).

 (File "2020-01-20-16-06-prediction_viewer_results.zip"):


#For results_log
#####Explanations of columns in results_log table (prediction results with location info)

* "prediction_idx": index of current predictions (0.. 6.5 millions)
* "epoch_num": prediction epoch, 0...1015 (each has 6400 predictions)
* "truth": encoding of the ground truth 
* "prediction": encoding of the prediction 
* "new_prediction": summary of the result by Max, with:
    * 0: pred == hogID
    * 1: pred == unkID
    * 2: pred == label (= truth)
    * 3: otherwise
* "file_id", "src_line", "ast_node_idx": location of the ast node


# Encoding of truth and prediction in results_log
##### Explanations of meaning of encodings for columns truth and prediction in results_log 

# Summary:
If *value* is ...

* 0: "EmptY", means ast node has no terminal value.
* 1..999: ID in the terminal dictionary.
* 1000: unk_id; this means that pointer network CANNOT predict (not in dict, 
		not in attention window), AND phog predicted wrongly.
* 1001: hog; this means that hog predicted correctly BUT pointer net failed completely.
* 1002: eof; which (probably) is the constant for the padding in preprocessing
* 1003...1053: token is can be found in the attention windown, with
 *  location_index = number of tokens to go back to get replica of current token = (*value*-(1003)+1). 
 * > Example: dic_value = 'TemplateSpec', i = 7 (node_idx), att_que {7} = [.., 'TemplateSpec', 'Simple', 'EmptY']. 
   > We have location_index = 3 and <value> = 1005 (in code *value* = location_id).  

### Explanation how we get this:

Infos from preprocess_code.get_terminal_extended.process():
* [line: 30]  unk_id: first ID that is not in terminal_dict
* [line: 48]  hog = unk_id + 1
* [line: 74]  order is [<seq_dict>] [unk, hog_id, eof] [<loc_idx>]
* [line: 74]  location_id = unk_id + 2 + location_index  # [unk, hog_id, eof, loc_idx]
* [line: 141] attn_size = 50
* [line: 148] process( ..., unk_id=vocab_size, ...)
* [line: 48]  location_index = [len(attn_que)-ind for ind, x
* [line: 48]      in enumerate(attn_que) if x == dic_value][-1]


Terminal dictionary (terminal_dict() is loaded from terminal_dict_filename.
*    We use "1k" size (effectively 1000).
*    Index 0 belongs to "EmptY" which means that the node has a non-terminal value.
*    => Effectively we have indices 1..999.

Used input files:
*    terminal_dict_filename = '../pickle_data/terminal_dict_1k_PY_train_dev.pickle'
    =>     vocab_size = unk_id = 1000
*    test_filename = '../../data/python10k_dev.json'
*    testHOG_filename = '../../data/phog_dev.json'

