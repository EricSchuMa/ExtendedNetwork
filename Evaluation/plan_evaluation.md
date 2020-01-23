# Brainstorming

## Possible options

### Data properties
+ (d10) ratio terminal to all nodes (not needed)
#### Principal power
+ How many times: 
  + (p05) RNN could have predicted? (i.e. #in_dict / #all)
  + (p06) attn could predict, i.e. ratio of cases when token in the attn windows
  + (p07) hog was correct, if selected?
  
### Predictions and selector decisions
Ratio of cases when
+ (p10) is the final result correct?
+ (p10e) while including empty?
+ (p20rnn) rnn prediction, and correct
+ (p20att) attn prediction, and correct
+ (p20hog) hog prediction, and correct

With current data we cannot do the following:
* (p50) were (RNN or att) AND PHOG correct together?
+ (p60) could be **any of the 3 methods** correct (independently of final pred)?
* (p70) were all 3 methods correct (together)?

 
#### Selector decisions
+ (p100) what is the split of prediction methods A. RNN, B. attn, C. phog
Ratio of cases when
* (p110) how many times when (p20) applies the prediction is correct (i.e. the selector was right)
+ (p120) PN have predicted and was wrong but phog would have been right?

## Distributions
### Files with prediction classes
Histogram of ratio of correct predicions per file.
I.e.:
+ For each file, compute ratio of correct predictions
+ Bin files by this ratio, plot histogram (or other distribution)
