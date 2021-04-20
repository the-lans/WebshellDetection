# Webshell Detection
Webshell Detection Based on the Word Attention Mechanism

Read [this article](https://github.com/the-lans/WebshellDetection/blob/main/Webshell%20Detection%20Based%20on%20the%20Word%20Attention%20Mechanism.pdf)

Cloned from https://github.com/leett1/Programe/

Editing directory  **project.zip**

Install packages:

- gensim==3.8.1
- python-Levenshtein==0.12.0
- pathlib==1.0.1
- numpy==1.19.2
- tensorflow==1.14
- keras==2.3.1
- scikit-learn==0.24.1



## **Initializing a machine learning environment**

### **Deploying a virtual environment Anaconda3**

- Install packages

  `yum install -y zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel xz xz-devel libffi-devel`

- Install Anaconda3

  `curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh`

  `bash Anaconda3-2019.03-Linux-x86_64.sh`

- Edit file .bashrc

  `vim  ~/.bashrc`

  Add this text to the end of the file:

  > export PATH="/home/<user>/anaconda3/bin:$PATH"

  `<user>` - replace with user

- `source ~/.bash_profile`

- Install environment

  `conda create -n tf1_env`

  `conda activate tf1_env`

 

### Project ML 

- `cd /var`

- `mkdir cnn_word2wec_sentence`

- `chmod 775 cnn_word2wec_sentence`

- `chown <user>:<group user> cnn_word2wec_sentence`

- `cd /var/cnn_word2wec_sentence`

  

### **Remote Python interpreter**

- Interpreter: `/home/<user>/anaconda3/envs/tf1_env/bin/python`

  `<user>` - replace with user

- Project migration: `/var/cnn_word2wec_sentence`

  

### **Python libraries**

- Project path

  `cd /var/cnn_word2wec_sentence`

- Activate env

  `conda activate tf1_env`

- Install TensorFlow

  `conda install tensorflow==1.14.0`

  `conda install keras==2.3.1`

- Install requirements

  `pip install -r requirements.txt`

- Check

  `python -V`

  `python -c 'import tensorflow as tf; print(tf.__version__)'`

  

## Train & test models

1. Model training: `python3 one_attention_model.py`

   Output: *one_attention_mode190626_dan.h5*

2. Model training: `python3 train_model.py`

   Output: *two_attention_mode190317.h5*

3. Model training: `python3 word2vec_train.py`

   Output: *word_train190313.model*

4. Edit the file and run the model test: `python3 test_1.py`

