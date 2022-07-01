# GPT (Generative Pre-trained Transformer)

We have 4 main files in this directory, namely, ```run_clm.py```, ```train_GPT2.ipynb```, ```songs_merged.txt``` and ```rihanna.txt```. The dataset was taken from [here](https://www.kaggle.com/datasets/paultimothymooney/poetry). All the song text files except ```rihanna.txt``` were merged into a single file (```songs_merged.txt```) for training and ```rihanna.txt``` was used for validation.

The ```run_clm.py``` code was obtained from [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py). It is recommended this file be run with TPU support in Google Colab.

Steps to run the the code and obtain the song:
- Upload the whole directory to the Google Colab Ubuntu workspace
- Run all cells of the ```train_GPT2.ipynb``` with TPU runtime enabled
- The model uses [perplexity](https://en.wikipedia.org/wiki/Perplexity) as the metric to evalute the performance
- The input prompt *I love deep learning* is already set in the code and the inference produces a beam-output of width 5. 
- Out of these outputs, the longest one is saved into the file ```final_song.txt```

You can also run train the GPT2 model directly by running the following command in the terminal:
```
python run_clm.py --model_type gpt2 \
                  --model_name_or_path gpt2 \
                  --train_file "songs_merged.txt" \
                  --validation_file "rihanna.txt" \
                  --per_device_train_batch_size 4 \
                  --per_device_eval_batch_size 4 \
                  --num_train_epochs 3 \
                  --output_dir "./gpt" \
                  --learning_rate 0.00001 \
```
But be mindful that this will only train the model and save the weights. You will need to write a separate function to load these weights and obtain predictions.