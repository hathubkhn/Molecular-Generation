#!/bin/bash

# task=classification  # target is a binary value (e.g., drug or not).
# dataset=hiv

task=regression  # target is a real value (e.g., energy eV).
dataset=pic50

radius=1
dim=50
layer_hidden=6
layer_output=6

batch_train=1024
batch_test=128
lr=1e-4
lr_decay=0.99
decay_interval=10
iteration=5000

setting=$dataset--radius$radius--dim$dim--layer_hidden$layer_hidden--layer_output$layer_output--batch_train$batch_train--batch_test$batch_test--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--iteration$iteration
python train.py $task $dataset $radius $dim $layer_hidden $layer_output $batch_train $batch_test $lr $lr_decay $decay_interval $weight_decay $iteration $setting
