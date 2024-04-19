# Kformer_code

To create embedding using Owl2Vec, use the github link: https://github.com/KRR-Oxford/OWL2Vec-Star and follow the Readme file. For this model, hermit axiom reasoner was used to created the embeddings with an vector size of 10 and 100. The embeddings can be found in the Owl2Vec_embeddings folder. 


Model Knowledge Injection

Use the github link: https://github.com/zjunlp/Kformer to get the code for Kformer. We have built the code for laptop domain. In the Model_injection folder, test_laptop.py can be used to run the task. The laptop folder needs to be placed in fairseq/examples/roberta as a folder so that the task can be registered. This code is unable to register the task with Roberta. gpt2_bpe has the dict file which is used by Roberta to pre-train the model. 

#this code can be used to pre-train the model in the terminal, you will need to update the folders accordingly.


TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8         # Adjusted number of sequences per batch (batch size for CPU)
UPDATE_FREQ=4           # Adjust as needed

DATA_DIR=.../data-bin

fairseq-train $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1

    
#to run the model, use this code


export LR=0.00001
export WARMUP_UPDATES=150
export MAX_UPDATES=100
export BSZ=16
export SEED=42
CUDA_VISIBLE_DEVICES=0 fairseq-train --ddp-backend=legacy_ddp \
    (file path to the databin) \
    --user-dir (path to the folder of the task) --restore-file (path to the best checkpoint) \
    --reset-optimizer --reset-dataloader --reset-meters --no-epoch-checkpoints --no-last-checkpoints \
    --no-save-optimizer-state --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sentiment_analysis --init-token 0 --bpe gpt2 --arch roberta_base --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --criterion sentence_ranking \
    --num-classes 3 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --batch-size $BSZ --max-update $MAX_UPDATES --log-format simple --log-interval 25 --seed $SEED \
    --save-dir (path where you want to save the model)
    --knowledge_layer 5 7


